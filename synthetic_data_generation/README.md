# Synthetic Data Generation Service

This service provides an API for generating synthetic data from locally trained
DoppelGANger models.

Model training is handled elsewhere: on demand from caller-supplied CSVs by the
separate `synthetic_model_training` service, or in batch from smart-meter
history by the standalone `services/synthetic_model_pretrainer` CLI. All of them
must point at the same `SYNTHETIC_MODEL_ROOT` folder or Docker volume, so a
model produced by any of them is immediately visible here without redeployment.

## Endpoints

### `GET /generate`

Generates `number_of_examples` synthetic sequences from a named model.

| parameter | meaning |
| --- | --- |
| `model_name` | model folder name |
| `user_id` | optional namespace (`models/<user_id>/<model_name>`) |
| `number_of_examples` | how many independent sequences to return |
| `hours` | length of each sequence; defaults to the length the model was trained on |
| `format` | `json` (default), `csv`, or `zip` of Parquet files |

Each example is an **independent draw**, not a segment of one longer series, and
their timestamps are a grid inherited from the training window rather than real
calendar dates.

`hours` is validated against the horizons the model was certified for (see
`capabilities` in its metadata) and rejected otherwise. The generator is
recurrent and will roll out to any length, so without this check the API would
silently serve output whose quality was never verified.

Values are clipped to the physical bounds recorded at training time
(`value_floor`/`value_ceiling`), since the generator's output layer is unbounded
and would otherwise emit negative average power for a counter that cannot run
backwards.

### `GET /models`

Lists available models with a capability summary for each: device, parameter,
resolution, days of real data trained on, and verified generation horizons.
Enough to build a model picker without one `/model-info` call per model.

Models trained before the capability evaluation have no summary; treat a missing
`offered_hours` as "only the trained length is known good".

### `GET /model-info`

Full training metadata for one model, including its capability evaluation.

### `GET /real-data`

The real series a model was trained on (`train_data.csv`), for comparing
generated output against it. Values are average power in W on the model's own
grid.

## Authentication

> [!IMPORTANT]
> Keycloak authentication is configured, but `auth.py` bypasses token
> verification for `GET` requests. Every endpoint in this service is a `GET`, so
> in practice **none of them require a token**.

## Features

-   **Synthetic Data Generation**: Generate realistic time-series data from pre-trained local models.
-   **Model Catalog**: List and inspect available local model folders.
-   **Configurable Requirements**: Switch between CPU and GPU dependencies via environment variables.

## Configuration

The service uses the following environment variable to determine which requirements to install and use:

- `SYNTHETIC_DATA_DEVICE`: Set to `CPU` (default) or `GPU`.

### Local Development

When running locally with `start_server.ps1`, the script will check this variable to install the appropriate `requirements.txt` or `requirementsGPU.txt`.

### Docker Deployment

The `docker-compose.yml` passes this variable as a build argument to the Dockerfile to ensure the correct dependencies are baked into the image.

## Running Locally

1. Uncomment the line in `start_server.ps1` that sets the device (optional):
   ```powershell
   $env:SYNTHETIC_DATA_DEVICE = "GPU"
   ```
2. Run the start script:
   ```powershell
   ./start_server.ps1
   ```

## Running with Docker (Standalone)

```powershell
./start_dockerized_server.ps1
```
