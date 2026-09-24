# Synthetic Data

The platform's synthetic data feature is split across three components. All of
them work with the same kind of model (DoppelGANger, a GAN for time series) and
share one model folder, `SYNTHETIC_MODEL_ROOT`. A model written there by either
trainer shows up in the generation API straight away, with no redeploy.

```text
 smart-meter data                      user CSV / JSON
 (data_provision)                            │
        │                                    ▼
        ▼                        synthetic_model_training (API)
 synthetic_model_pretrainer (CLI)            │
        │                                    │
        └──────────►  SYNTHETIC_MODEL_ROOT  ◄┘
                              │
                              ▼
               synthetic_data_generation (API)
                              │
                              ▼
                    synthetic time series
```

| Component | What it is | Runs as | Port (host → container) |
| --- | --- | --- | --- |
| [`synthetic_model_pretrainer`](synthetic_model_pretrainer/README.md) | Trains the shared models from real smart-meter data | Local CLI, not part of the stack | — |
| [`synthetic_model_training`](synthetic_model_training/README.md) | Trains a model on demand from a file the caller uploads | Docker service | `8007 → 601` |
| [`synthetic_data_generation`](synthetic_data_generation/README.md) | Serves synthetic data from any trained model | Docker service | `8004 → 600` |

## `synthetic_model_pretrainer`: shared models

An offline CLI that builds the models shipped with the platform. It fetches
readings from `data_provision` and trains one model per smart meter. Before
training it converts the cumulative energy counter into average power and
resamples every device to a 15-minute grid. It also includes tools to audit the
raw data (`audit_raw.py`) and to certify which generation lengths each model
handles well (`capabilities.py`).

Real device ids are kept in a git-ignored `.env`, never in tracked files.

## `synthetic_model_training`: on-demand training

A small API (`POST /train`) that trains a model on a CSV or JSON file exactly
as it is uploaded. Any preprocessing is up to the caller. Models trained this
way have no capability evaluation. The model code is not duplicated: the
Dockerfile copies it from `synthetic_data_generation/`.

## `synthetic_data_generation`: the user-facing API

Lists the available models (`GET /models`) and generates synthetic sequences
from them (`GET /generate`) as JSON, CSV or zipped Parquet. The main safeguards:

- **Pseudonymisation**: pre-trained models are published only as
  `ResidentialConsumer_NN`, their provenance metadata is stripped, and the
  timestamps of generated data start at the moment of generation.
- **Certified horizons**: `hours` is only accepted for lengths the model passed
  in `capabilities.py`.
- **Physical bounds**: output is clipped to the value range seen in training.

## Typical flow

1. Pretrain shared models with the CLI, or train your own through
   `synthetic_model_training`.
2. Make sure both trainers and the generation service use the same
   `SYNTHETIC_MODEL_ROOT`. In `docker-compose.yml` this is
   `./synthetic_model_pretrainer/models`, mounted at `/app/models`.
3. Call `synthetic_data_generation` to list models and generate data.

For setup, parameters and details, see each component's own README.
