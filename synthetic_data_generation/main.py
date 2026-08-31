import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Response,
    status,
)
import zipfile
import io

from dgan_wrapper import generate, load_train_result
from fastapi.responses import JSONResponse
from auth import KEYCLOAK_CLIENT_ID
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Synthetic Data API",
    description="API for generating synthetic time-series data from locally pre-trained models.",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    swagger_ui_init_oauth={
        "clientId": KEYCLOAK_CLIENT_ID,
        "appName": "Synthetic Data Generation",
    },
    # No app-level auth dependency: every route below is GET, and verify_token
    # already bypasses GET requests on its own (see auth.py). An app-level
    # dependency here would also apply to the /control router mounted below,
    # which the orchestrator worker calls without a Keycloak token.
)

# Add CORS middleware to allow Swagger UI to communicate with Keycloak
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI-Effect Control Interface ---
try:
    from common import create_control_router, synthetic_data_generation_handlers

    app.include_router(
        create_control_router(synthetic_data_generation_handlers),
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


def _model_path(model_name: str, user_id: Optional[str] = None) -> Path:
    if user_id:
        return _safe_child_path(user_id, model_name)

    direct_path = _safe_child_path(model_name)
    if (direct_path / "model.pt").exists():
        return direct_path

    matches = [
        path.parent
        for path in MODEL_ROOT.rglob("model.pt")
        if path.parent.name == model_name
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple pre-trained models named '{model_name}' exist. "
                "Provide user_id or use a unique model folder name."
            ),
        )
    return direct_path


def _value_bounds(model_dir: Path) -> tuple[Optional[float], Optional[float]]:
    """Physical bounds recorded at training time, if any.

    The generator's output is unbounded, so without these it emits negative
    average power for an import register that cannot run backwards.
    """
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return None, None
    try:
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError):
        return None, None
    return metadata.get("value_floor"), metadata.get("value_ceiling")


def _resolve_horizon_steps(model_dir: Path, train_result, hours: int) -> int:
    """Timesteps for a requested horizon, refused unless the model was measured
    to support it.

    The generator will happily roll out to any length, so without this check the
    API would silently serve output whose quality was never verified.
    """
    metadata_path = model_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, ValueError):
            metadata = {}
    caps = metadata.get("capabilities") or {}

    # resolution_minutes is a top-level training field; capabilities repeats it.
    resolution = metadata.get("resolution_minutes") or caps.get("resolution_minutes")
    if not resolution:
        trained_hours = caps.get("trained_sequence_hours")
        if trained_hours:
            resolution = trained_hours * 60 // int(train_result.model.max_sequence_len)
    if not resolution:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This model has no recorded resolution, so 'hours' cannot be "
            "translated into timesteps. Omit 'hours' to use the trained length.",
        )

    offered = caps.get("offered_hours")
    if not offered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This model has no evaluated generation horizons. Omit 'hours' "
            "to use the length it was trained on.",
        )
    if hours not in offered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_dir.name}' does not support a {hours}h horizon. "
            f"Supported: {offered}.",
        )
    return hours * 60 // resolution


@app.get("/generate", tags=["Synthetic Data Generation"])
def generate_synthetic_data(
    model_name: str = Query("test_model", description="The name of the model"),
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
    number_of_examples: int = Query(1, description="Number of examples to generate"),
    hours: Optional[int] = Query(
        None,
        description="Length of each generated example, in hours. Defaults to the "
        "length the model was trained on. Only values listed in the model's "
        "capabilities.offered_hours are accepted - longer rollouts are possible "
        "architecturally but were measured to degrade.",
    ),
    format: str = Query(
        "json",
        pattern="^(json|csv|zip)$",
        description="Result format: 'json' (default, list of records), "
        "'csv' (single CSV attachment), or 'zip' (ZIP of Parquet files).",
    ),
):
    """
    Endpoint to generate synthetic data using a locally pre-trained DGAN model.
    """
    folder_path = _model_path(model_name, user_id)
    try:
        train_result = load_train_result(str(folder_path))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pre-trained model '{model_name}' not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {e}",
        )

    horizon_steps = None
    if hours is not None:
        horizon_steps = _resolve_horizon_steps(folder_path, train_result, hours)

    floor, ceiling = _value_bounds(folder_path)
    synthetic_dfs = generate(
        train_result,
        num_examples=number_of_examples,
        horizon_steps=horizon_steps,
        clip_min=floor,
        clip_max=ceiling,
    )

    if not synthetic_dfs:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    for i, synthetic_df in enumerate(synthetic_dfs):
        synthetic_df["example_id"] = i + 1
    combined_df = pd.concat(synthetic_dfs, ignore_index=True)

    if format == "json":
        # combined_df holds numpy dtypes (int64/float64) that plain json.dumps
        # can't serialize; to_json() handles that conversion natively.
        records = json.loads(combined_df.to_json(orient="records", date_format="iso"))
        return JSONResponse(content={"model_name": model_name, "data": records})
    elif format == "csv":
        return Response(
            combined_df.to_csv(index=False),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=synthetic_data_{model_name}.csv"
            },
        )
    else:
        zip_io = io.BytesIO()
        with zipfile.ZipFile(
            zip_io, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as temp_zip:
            for i, synthetic_df in enumerate(synthetic_dfs):
                zip_path = f"synthetic_data_{i+1}.parquet"
                temp_zip.writestr(zip_path, synthetic_df.to_parquet(index=False))
        return Response(
            zip_io.getvalue(),
            media_type="application/x-zip-compressed",
            headers={
                "Content-Disposition": f"attachment; filename=synthetic_data_{model_name}.zip"
            },
        )


def _capability_summary(model_dir: Path) -> dict:
    """The few metadata fields a model picker needs, read from metadata.json."""
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError):
        return {}
    caps = metadata.get("capabilities") or {}
    return {
        "device_id": metadata.get("device_id"),
        "parameter": metadata.get("parameter"),
        "resolution_minutes": metadata.get("resolution_minutes"),
        "trained_at": metadata.get("trained_at"),
        "trained_sequence_hours": caps.get("trained_sequence_hours"),
        "real_days_trained_on": caps.get("real_days_trained_on"),
        "offered_hours": caps.get("offered_hours"),
        "max_offered_hours": caps.get("max_offered_hours"),
        "capabilities_evaluated_at": caps.get("evaluated_at"),
    }


@app.get("/models", tags=["Synthetic Data Generation"])
async def list_available_trained_models(
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
):
    user_models_dir = _safe_child_path(user_id) if user_id else MODEL_ROOT
    available_models = []
    if user_models_dir.exists():
        for model_path in sorted(user_models_dir.rglob("model.pt")):
            model_dir = model_path.parent
            entry = {
                "name": model_dir.name,
                "namespace": str(model_dir.parent.relative_to(MODEL_ROOT))
                if model_dir.parent != MODEL_ROOT
                else None,
            }
            # Summarise what this model can actually do, so a client can build a
            # picker without one /model-info round-trip per model. Absent for
            # models that predate the capability evaluation - clients should
            # treat a missing summary as "only the trained length is known good".
            entry.update(_capability_summary(model_dir))
            available_models.append(entry)
    return available_models


@app.get("/real-data", tags=["Synthetic Data Generation"])
async def get_real_training_data(
    model_name: str = Query("test_model", description="The name of the model"),
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
    max_points: int = Query(
        5000, ge=96, le=50000, description="Cap on returned points; the series is truncated, not resampled."
    ),
) -> dict:
    """The real series a model was trained on, for comparison against its output.

    Served from train_data.csv, which the pretrainer writes next to the model.
    Values are average power in W on the model's own grid - already converted
    from the raw cumulative counter, so they are directly comparable with
    generated output.
    """
    folder_path = _model_path(model_name, user_id)
    train_csv = folder_path / "train_data.csv"
    if not train_csv.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training data stored for model '{model_name}'.",
        )
    frame = pd.read_csv(train_csv)
    value_columns = [c for c in frame.columns if c != "datetime"]
    if not value_columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="train_data.csv has no value column.",
        )
    column = value_columns[0]
    resolution = None
    metadata_path = folder_path / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open(encoding="utf-8") as f:
                resolution = json.load(f).get("resolution_minutes")
        except (OSError, ValueError):
            resolution = None

    frame = frame.head(max_points)
    return {
        "model": folder_path.name,
        "column": column,
        "resolution_minutes": resolution,
        "n_points": int(len(frame)),
        "start": str(frame["datetime"].iloc[0]) if "datetime" in frame and len(frame) else None,
        "values": [float(v) for v in frame[column].to_numpy()],
    }


@app.get("/model-info", tags=["Synthetic Data Generation"])
async def get_model_info(
    model_name: str = Query("test_model", description="The name of the model"),
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
) -> dict:
    folder_path = _model_path(model_name, user_id)
    if not (folder_path / "model.pt").exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pre-trained model '{model_name}' not found",
        )

    metadata_path = folder_path / "metadata.json"
    info = {
        "name": folder_path.name,
        "namespace": str(folder_path.parent.relative_to(MODEL_ROOT))
        if folder_path.parent != MODEL_ROOT
        else None,
    }
    if metadata_path.exists():

        with metadata_path.open("r", encoding="utf-8") as f:
            info["metadata"] = json.load(f)
    return info


if __name__ == "__main__":
    port = int(os.getenv("ML_GRETEL_PORT", 600))
    reload = os.getenv("ML_GRETEL_RELOAD", "True").lower() == "true"

    print(f"Starting server. Go to http://127.0.0.1:{port}/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
