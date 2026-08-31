"""TEF operation handlers for synthetic_data_generation service.

Exposes three operations to the AI-Effect orchestrator, each a self-call to
this service's own GET endpoints. All three stay public even when
AUTH_ENABLED=True because auth.py only enforces the Keycloak check on
non-GET requests (see main.py / auth.py), so no service token is needed -
same approach data_provision uses for its FetchData handler.

  - GenerateData: generate synthetic examples from a pre-trained model
  - ListModels:   list available pre-trained models
  - GetModelInfo: get metadata for a single pre-trained model

Handler inputs are passed as inline base64-encoded JSON by the orchestrator.
"""

import base64
import json
import logging

import httpx

from .control_interface import DataReference, ExecuteRequest, ExecuteResponse, get_data_url
from .task_manager import get_task_manager

logger = logging.getLogger(__name__)

_SELF = "http://localhost:600"


def _decode_params_ref(input_ref: dict) -> dict:
    """Decode a DataReference's JSON payload regardless of protocol.

    Standalone use passes params as an inline base64 blob at workflow
    submission time. Chained after another node (e.g. synthetic_model_
    training.TrainModel, whose metadata output is exactly this kind of
    params-shaped JSON), the input instead arrives as an "http" reference
    that has to be fetched first.
    """
    protocol = input_ref.get("protocol")
    try:
        if protocol == "inline":
            return json.loads(base64.b64decode(input_ref.get("uri", "")).decode())
        if protocol == "http":
            with httpx.Client(timeout=30.0) as client:
                response = client.get(input_ref["uri"])
            response.raise_for_status()
            return response.json()
    except Exception:
        return {}
    return {}


def _params_from_inputs(request: ExecuteRequest) -> dict:
    if request.inputs:
        return _decode_params_ref(request.inputs[0])
    return {}


def _namespace_from(params: dict) -> str | None:
    """synthetic_model_training calls this 'namespace'; this service still
    calls the identical concept 'user_id' - accept either so a model trained
    upstream chains straight into a downstream GenerateData/ListModels/
    GetModelInfo call without the two services needing to agree on a name."""
    return params.get("namespace") or params.get("user_id")


def _self_get(endpoint: str, params: dict, request: ExecuteRequest) -> ExecuteResponse:
    url = f"{_SELF}{endpoint}"
    logger.info(f"{request.method}: GET {url} params={params}")

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(url, params=params)

        if response.status_code != 200:
            return ExecuteResponse(
                status="failed",
                error=f"Request failed ({response.status_code}): {response.text}",
            )

        content_type = response.headers.get("content-type", "")
        if "text/csv" in content_type:
            fmt = "csv"
        elif "zip" in content_type:
            fmt = "zip"
        else:
            fmt = "json"

        get_task_manager().store_data(request.task_id, response.content, fmt)

        return ExecuteResponse(
            status="complete",
            output=DataReference(
                protocol="http",
                uri=get_data_url(request.task_id),
                format=fmt,
            ),
        )
    except Exception as e:
        logger.error(f"{request.method} failed: {e}")
        return ExecuteResponse(status="failed", error=str(e))


def execute_GenerateData(request: ExecuteRequest) -> ExecuteResponse:
    """Generate synthetic examples from a pre-trained model.

    Input (inline JSON):
        model_name: str (default "test_model")
        user_id: str (optional namespace)
        number_of_examples: int (default 1)
        format: "json" | "csv" | "zip" (default "json")
    """
    params = _params_from_inputs(request)
    query_params = {
        "model_name": params.get("model_name", "test_model"),
        "number_of_examples": params.get("number_of_examples", 1),
        "format": params.get("format", "json"),
    }
    namespace = _namespace_from(params)
    if namespace:
        query_params["user_id"] = namespace

    return _self_get("/generate", query_params, request)


def execute_ListModels(request: ExecuteRequest) -> ExecuteResponse:
    """List available pre-trained models.

    Input (inline JSON):
        user_id: str (optional namespace)
    """
    params = _params_from_inputs(request)
    query_params = {}
    namespace = _namespace_from(params)
    if namespace:
        query_params["user_id"] = namespace

    return _self_get("/models", query_params, request)


def execute_GetModelInfo(request: ExecuteRequest) -> ExecuteResponse:
    """Get metadata for a single pre-trained model.

    Input (inline JSON):
        model_name: str (default "test_model")
        user_id: str (optional namespace)
    """
    params = _params_from_inputs(request)
    query_params = {"model_name": params.get("model_name", "test_model")}
    namespace = _namespace_from(params)
    if namespace:
        query_params["user_id"] = namespace

    return _self_get("/model-info", query_params, request)


synthetic_data_generation_handlers = {
    "GenerateData": execute_GenerateData,
    "ListModels": execute_ListModels,
    "GetModelInfo": execute_GetModelInfo,
}
