"""TEF operation handlers for synthetic_model_training service.

Exposes one operation to the AI-Effect orchestrator:
  - TrainModel: train a DoppelGANger model from a data file

Training is triggered in-process (calling dgan_wrapper.train directly)
rather than via a self-call to POST /train, because /train is a POST
endpoint, which is Keycloak-guarded when AUTH_ENABLED=True (only GET
requests bypass auth - see auth.py). A plain HTTP self-call would get a
401; calling the training function directly sidesteps that without needing
a service token, the same reasoning as knowledge_store's control interface.

Training takes minutes, so unlike the other services' handlers this one
runs asynchronously: it registers the task as "running" and hands the
actual work to a background thread, returning immediately. The orchestrator
worker then polls /control/status/{task_id} until the training thread marks
it complete or failed (see orchestrator's services/worker.py -
_poll_until_complete). Note this handler registers the task itself before
returning "running", so the router's own post-handler
task_manager.register() call (see control_interface.create_control_router)
just re-applies the same "running"/progress=0 state - harmless, since real
training takes far longer than that window to report its first progress
update.

Handler inputs are passed as inline base64-encoded JSON by the orchestrator.
"""

import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx
import pandas as pd

from dgan_wrapper import save_train_result, train

from .control_interface import DataReference, ExecuteRequest, ExecuteResponse, get_data_url
from .task_manager import get_task_manager

logger = logging.getLogger(__name__)

MODEL_ROOT = Path(os.getenv("SYNTHETIC_MODEL_ROOT", "models")).resolve()


def _safe_model_path(model_name: str, namespace: str | None) -> Path:
    path = MODEL_ROOT.joinpath(namespace, model_name) if namespace else MODEL_ROOT / model_name
    path = path.resolve()
    if MODEL_ROOT != path and MODEL_ROOT not in path.parents:
        raise ValueError("Invalid model path")
    return path


def _resolve_data_reference(ref: dict) -> tuple[bytes, str]:
    protocol = ref.get("protocol")
    fmt = ref.get("format", "csv")
    uri = ref.get("uri", "")

    if protocol == "inline":
        return base64.b64decode(uri), fmt
    if protocol == "http":
        with httpx.Client(timeout=60.0) as client:
            response = client.get(uri)
        response.raise_for_status()
        return response.content, fmt
    if protocol == "file":
        with open(uri, "rb") as f:
            return f.read(), fmt

    raise ValueError(f"Unsupported data protocol: {protocol!r}")


def _bytes_to_df(content: bytes, extension: str) -> pd.DataFrame:
    if extension == "csv":
        return pd.read_csv(io.BytesIO(content))
    if extension == "json":
        parsed = json.loads(content.decode("utf-8"))
        if isinstance(parsed, dict) and "data" in parsed:
            return pd.DataFrame(parsed["data"])
        return pd.read_json(io.StringIO(content.decode("utf-8")))
    raise ValueError(f"Unsupported data format for training: {extension!r}")


def _split_inputs(inputs: list[dict]) -> tuple[dict | None, dict]:
    """Split task inputs into (data_reference, parameters)."""
    data_ref = None
    params: dict = {}

    for inp in inputs:
        if inp.get("protocol") == "inline" and inp.get("format") == "json":
            decoded = json.loads(base64.b64decode(inp.get("uri", "")).decode())
            if isinstance(decoded.get("data"), dict):
                data_ref = decoded.pop("data")
            params = decoded
        elif data_ref is None:
            data_ref = inp

    return data_ref, params


def _run_training(
    task_id: str,
    train_df: pd.DataFrame,
    folder_path: Path,
    model_name: str,
    namespace: str | None,
    index_col: str,
    sequence_len: int,
    sample_len: int,
    batch_size: int,
    epochs: int,
) -> None:
    task_manager = get_task_manager()

    def _progress_callback(progress_info) -> None:
        total_steps = max(progress_info.total_epochs * progress_info.total_batches, 1)
        done_steps = progress_info.epoch * progress_info.total_batches + progress_info.batch + 1
        task_manager.update_progress(task_id, int(100 * done_steps / total_steps))
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            with (folder_path / "progress.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": progress_info.epoch + 1,
                        "total_epochs": progress_info.total_epochs,
                        "batch": progress_info.batch + 1,
                        "total_batches": progress_info.total_batches,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    },
                    f,
                )
        except Exception:
            logger.exception("Failed to write progress.json")

    try:
        train_result = train(
            train_df,
            index_col=index_col or None,
            sequence_len=sequence_len,
            sample_len=sample_len,
            batch_size=batch_size,
            epochs=epochs,
            progress_callback=_progress_callback,
        )
        save_train_result(train_result, str(folder_path))

        metadata = {
            "model_name": model_name,
            "namespace": namespace,
            "index_col": index_col or None,
            "sequence_len": sequence_len,
            "sample_len": sample_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with (folder_path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        task_manager.store_data(task_id, json.dumps(metadata).encode(), "json")
        task_manager.complete(
            task_id,
            DataReference(protocol="http", uri=get_data_url(task_id), format="json"),
        )
    except Exception as e:
        logger.error(f"TrainModel failed: {e}")
        task_manager.fail(task_id, str(e))


def execute_TrainModel(request: ExecuteRequest) -> ExecuteResponse:
    """Train a DoppelGANger model from a data file. Runs asynchronously.

    Parameters (inline JSON):
        model_name: str (default "test_model")
        namespace: str (optional)
        overwrite: bool (default False)
        index_col: str (default "datetime")
        sequence_len: int (default 100)
        sample_len: int (default 10)
        batch_size: int (default 1000)
        epochs: int (default 10)
        data: nested DataReference to the training file - only needed when
            this node has no upstream data-producing predecessor.
    """
    data_ref, params = _split_inputs(request.inputs)
    if data_ref is None:
        return ExecuteResponse(status="failed", error="No data input provided")

    model_name = params.get("model_name", "test_model")
    namespace = params.get("namespace")
    overwrite = params.get("overwrite", False)

    try:
        folder_path = _safe_model_path(model_name, namespace)
    except ValueError as e:
        return ExecuteResponse(status="failed", error=str(e))

    if folder_path.exists() and not overwrite:
        return ExecuteResponse(
            status="failed",
            error=f"Model {model_name} already exists and overwrite is set to False",
        )

    try:
        content, extension = _resolve_data_reference(data_ref)
        train_df = _bytes_to_df(content, extension)
    except Exception as e:
        return ExecuteResponse(status="failed", error=f"Error reading training data: {e}")

    task_manager = get_task_manager()
    task_manager.register(request.task_id, status="running", progress=0)

    thread = threading.Thread(
        target=_run_training,
        args=(
            request.task_id,
            train_df,
            folder_path,
            model_name,
            namespace,
            params.get("index_col", "datetime"),
            params.get("sequence_len", 100),
            params.get("sample_len", 10),
            params.get("batch_size", 1000),
            params.get("epochs", 10),
        ),
        daemon=True,
    )
    thread.start()

    return ExecuteResponse(status="running", task_id=request.task_id)


synthetic_model_training_handlers = {
    "TrainModel": execute_TrainModel,
}
