# MIT License

Copyright (C) 2026, INESC TEC

**consumer-data-synthetization**: Generates privacy-preserving synthetic energy time series from generative models (DoppelGANger) trained on real smart-meter data, exposed through a REST API.

## Authors

This software is authored by:

- Alexandre Costa
- Olga Klyagina
- Carlos Santos
- Carlos Silva

## Third-party libraries

This program makes use and is distributed with the following libraries:

- FastAPI (<https://github.com/fastapi/fastapi>)
- Uvicorn[standard] (<https://github.com/encode/uvicorn>)
- python-multipart (<https://github.com/Kludex/python-multipart>)
- pydantic (<https://github.com/pydantic/pydantic>)
- torch (PyTorch) (<https://github.com/pytorch/pytorch>)
- tensorboard (<https://github.com/tensorflow/tensorboard>)
- pandas (<https://github.com/pandas-dev/pandas>)
- numpy (<https://github.com/numpy/numpy>)
- httpx (<https://github.com/encode/httpx>)
- requests (<https://github.com/psf/requests>)
- python-dotenv (<https://github.com/theskumar/python-dotenv>)
- matplotlib (<https://github.com/matplotlib/matplotlib>)
- python-keycloak (<https://github.com/marcospereirampj/python-keycloak>)
- doppelganger-torch (Gretel) (<https://github.com/gretelai/doppelganger-torch>)
- Python (python:3.11-slim base image) (<https://github.com/python/cpython>)

## Changes to the doppelganger-torch (Gretel) library

Our version of `doppelganger.py` is derived from the Gretel doppelganger-torch project. Ignoring comments and docstrings, 31 of its 40 code units are identical to the original; 9 were modified, 5 were added, and none were removed. The changes are the following.

**Normalisation of constant series.** The original divides by the range of the training data when scaling it, which fails when a meter reports the same value for the whole period. This produced invalid numbers instead of a usable model. Our version clamps the range to a small non-zero value first, so a flat series trains to a degenerate but well-defined model rather than failing.

**Deterministic generation.** The original runs the generator without switching it out of training mode, so its batch normalisation layers adapt to whatever random batch is being generated. The output for a given input therefore depended on what else was generated alongside it. Our version switches the generator to inference mode around generation and restores the previous state afterwards, which makes repeated generation reproducible and allows the diagnostic tools to fix a random seed.

**Type checking.** The original decides how to treat each output by comparing the text of its class name. Our version uses a proper type check instead, in the four places where this occurs.

**Variable-length generation.** The original can only produce sequences of the single length the model was trained on. We added a new method that drives the generator for any multiple of its internal block size, so a model trained on one day can produce several days. This is what allows the service to offer a choice of sequence length, and to verify the output quality separately at each one. Only the discriminator is fixed-width, and it is used during training only.

**Model saving.** The original defines its main class as a plain Python class and builds part of it from anonymous functions, neither of which can be saved to disk by the standard PyTorch mechanism. Our version derives the class from the PyTorch module base class and replaces the anonymous functions with ordinary methods, storing the values they need as attributes. This is what makes a trained model loadable in a separate process.

**Progress reporting.** We added an optional callback to the training routine, invoked once per batch with the current epoch and batch number, so a long training run can report progress to the calling service.

**Removal of development instrumentation.** The original writes extensive diagnostic output during training: a computation graph dump, activation and gradient histograms, per-batch summaries, console messages and a progress bar. All of this was removed, since the service runs training unattended.

## Contact

You can reach INESC TEC Technology Transfer Office (TTO) at <tech-transfer@inesctec.pt>, or

Campus da Faculdade de Engenharia da Universidade do Porto  
Rua Dr. Roberto Frias  
4200-465 Porto  
Portugal

## License

Licensed under the terms of MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
