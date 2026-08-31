# Synthetic Model Training Service

This service trains DoppelGANger models on demand from data the caller supplies,
for the Synthetic Data Generation service to serve.

It is deployed separately from the user-facing generation API. Both services
must share the same `SYNTHETIC_MODEL_ROOT` folder or Docker volume.

The model code (`dgan_wrapper.py`, `doppelganger.py`) is not duplicated here: the
Dockerfile copies it from `synthetic_data_generation/`, so there is a single
source of truth for both services.

## Endpoints

### `POST /train`

Trains a new model from an uploaded CSV or JSON file.

| parameter | default | meaning |
| --- | --- | --- |
| `uploaded_file` | required | training data (CSV or JSON) |
| `model_name` | `test_model` | name of the model folder |
| `namespace` | none | optional scope, e.g. a demographic or user id |
| `index_col` | `datetime` | timestamp column name |
| `sequence_len` | 100 | timesteps per training sequence |
| `sample_len` | 10 | timesteps emitted per recurrent step; must divide `sequence_len` |
| `batch_size` | 1000 | training batch size |
| `epochs` | 10 | training epochs |
| `overwrite` | false | replace an existing model of the same name |

`user_id` is a deprecated alias for `namespace`, kept for compatibility.

### `GET /models`

Lists models trained through this API.

### `GET /training_info`

Returns a given model's training metadata.

## Relationship to the offline pretrainer

This API and `services/synthetic_model_pretrainer` produce models in the same
format, but they are **not equivalent** in what they do to the data:

- This service trains on the CSV as given. Whatever preprocessing the data needs
  is the caller's responsibility.
- The offline pretrainer owns a full pipeline: it fetches raw smart-meter
  readings, converts a cumulative energy counter into average power, resamples
  every device onto a common 15-minute grid, handles nulls and gaps, filters
  physically impossible readings, and aligns sequences to calendar days.

Models trained here therefore carry no `capabilities` block, and the generation
API will not accept an `hours` parameter for them.

## Model layout

```text
models/<model_name>/
models/<namespace>/<model_name>/
```

The generation service reads the same folders.

## Authentication

`POST /train` requires a valid Keycloak token when `AUTH_ENABLED` is set. Note
that `auth.py` bypasses verification for `GET` requests, so `/models` and
`/training_info` are served without one.
