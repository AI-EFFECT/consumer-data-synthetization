import io
import json
import logging
import os
import time
from enum import StrEnum
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from auth import KEYCLOAK_CLIENT_ID, verify_token
from dgan_wrapper import save_train_result, train

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "False").lower() == "true"

app = FastAPI(
    title="Synthetic Model Training API",
    description="API for training local models used by the Synthetic Data Generation service.",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    swagger_ui_init_oauth={
        "clientId": KEYCLOAK_CLIENT_ID,
        "appName": "Synthetic Model Training",
    },
    # No app-level auth dependency: it would also apply to the /control
    # router mounted below, which the orchestrator worker calls without a
    # Keycloak token. verify_token is instead applied directly to /train,
    # the one route below that actually needs it (GET routes already bypass
    # verify_token on their own - see auth.py).
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI-Effect Control Interface ---
try:
    from common import create_control_router, synthetic_model_training_handlers

    app.include_router(
        create_control_router(synthetic_model_training_handlers),
        prefix="/control",
    )
except ImportError as e:
    logger.warning(f"AI-Effect control interface not available: {e}")

MODEL_ROOT = Path(os.getenv("SYNTHETIC_MODEL_ROOT", "models")).resolve()


def _safe_child_path(*parts: str) -> Path:
    path = MODEL_ROOT.joinpath(*parts).resolve()
    if MODEL_ROOT != path and MODEL_ROOT not in path.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model path",
        )
    return path


def _model_path(model_name: str, namespace: str | None = None) -> Path:
    return _safe_child_path(namespace, model_name) if namespace else _safe_child_path(model_name)


class SupportedFileExtensions(StrEnum):
    """File extensions accepted for training data uploads."""

    CSV = "csv"
    JSON = "json"


def _read_uploaded_file_as_df(file: UploadFile) -> pd.DataFrame:
    """Parse an uploaded training file into a DataFrame, inferring the format
    from its filename extension so callers don't pass a separate format param."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename not provided.")

    extension_string = file.filename.rsplit(".", 1)[-1].strip().lower()
    try:
        extension = SupportedFileExtensions(extension_string)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '.{extension_string}'. "
                f"Supported: {', '.join(e.value for e in SupportedFileExtensions)}"
            ),
        )

    contents = file.file.read()

    if extension == SupportedFileExtensions.CSV:
        return pd.read_csv(io.BytesIO(contents))

    parsed_json = json.loads(contents.decode("utf-8"))
    if isinstance(parsed_json, dict) and "data" in parsed_json:
        return pd.DataFrame(parsed_json["data"])
    return pd.read_json(io.StringIO(contents.decode("utf-8")))


@app.post(
    "/train",
    tags=["Synthetic Model Training"],
    dependencies=[Depends(verify_token)] if AUTH_ENABLED else [],
)
async def train_new_model(
    uploaded_file: UploadFile = File(..., description="CSV or JSON file for training data"),
    model_name: str = Query("test_model", description="The name of the model"),
    namespace: str | None = Query(
        None,
        description="Optional namespace, for example demographic or test_user_id.",
    ),
    user_id: str | None = Query(
        None,
        description="Deprecated alias for namespace. Kept for compatibility.",
    ),
    index_col: str = Query("datetime", description="The name of the index column"),
    overwrite: bool = Query(False, description="Whether to overwrite existing model"),
    sequence_len: int = Query(100, description="Sequence length for training"),
    sample_len: int = Query(
        10, description="Internal sample length, must be a divisor of sequence_len"
    ),
    batch_size: int = Query(1000, description="Batch size for training"),
    epochs: int = Query(10, description="Number of epochs for training"),
) -> Response:
    namespace = namespace or user_id
    train_df = _read_uploaded_file_as_df(uploaded_file)
    folder_path = _model_path(model_name, namespace)

    if folder_path.exists() and not overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {model_name} already exists and overwrite is set to False",
        )

    def _progress_callback(progress_info):
        info = {
            "epoch": progress_info.epoch + 1,
            "total_epochs": progress_info.total_epochs,
            "batch": progress_info.batch + 1,
            "total_batches": progress_info.total_batches,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        logger.info(info)
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            with (folder_path / "progress.json").open("w", encoding="utf-8") as f:
                json.dump(info, f)
        except Exception as e:
            logger.exception(e)

    logger.info("Starting model training...")
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
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during training setup: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during training: {e}",
        )

    logger.info("Saving model...")
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

    return Response(
        f"Model {model_name} trained successfully.", status_code=status.HTTP_201_CREATED
    )


@app.get("/models", tags=["Synthetic Model Training"])
async def list_available_trained_models(
    namespace: str | None = Query(None, description="Optional model namespace"),
    user_id: str | None = Query(
        None,
        description="Deprecated alias for namespace. Kept for compatibility.",
    ),
):
    namespace = namespace or user_id
    models_dir = _safe_child_path(namespace) if namespace else MODEL_ROOT
    available_models = []
    if models_dir.exists():
        for model_path in models_dir.rglob("model.pt"):
            model_dir = model_path.parent
            available_models.append(
                {
                    "name": model_dir.name,
                    "namespace": str(model_dir.parent.relative_to(MODEL_ROOT))
                    if model_dir.parent != MODEL_ROOT
                    else None,
                }
            )
    return available_models


@app.get("/training_info", tags=["Synthetic Model Training"])
async def get_training_info(
    model_name: str = Query("test_model", description="The name of the model"),
    namespace: str | None = Query(None, description="Optional model namespace"),
    user_id: str | None = Query(
        None,
        description="Deprecated alias for namespace. Kept for compatibility.",
    ),
) -> dict:
    namespace = namespace or user_id
    training_info_path = _model_path(model_name, namespace) / "progress.json"
    if not training_info_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training info found for model '{model_name}'",
        )
    with training_info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    port = int(os.getenv("SYNTHETIC_TRAINING_PORT", 6001))
    reload = os.getenv("SYNTHETIC_TRAINING_RELOAD", "True").lower() == "true"

    print(f"Starting server. Go to http://127.0.0.1:{port}/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
