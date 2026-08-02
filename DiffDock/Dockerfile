##The 3-Clause BSD License
#
## Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

ARG PYG_ARCHIVE=DiffDock/maca-pyg-2.7.0-torch2.8-py312-3.8.0.11-ubuntu24.04-amd64.tar.xz
ARG DIFFDOCK_SOURCE=DiffDock/DiffDock

FROM cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.8-py312-ubuntu24.04-amd64

ARG PYG_ARCHIVE
ARG DIFFDOCK_SOURCE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/root/.cache/torch \
    DIFFDOCK_WEIGHTS=/weights/diffdock

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    ca-certificates \
    build-essential \
    libmetis-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    numpy \
    pandas \
    scipy \
    pyyaml \
    tqdm \
    requests \
    networkx \
    scikit-learn \
    pybind11 \
    rdkit \
    prody \
    biopython \
    aiohttp \
    fsspec \
    jinja2 \
    psutil \
    pyparsing \
    xxhash

COPY ${PYG_ARCHIVE} /tmp/maca-pyg.tar.xz

RUN mkdir -p /tmp/maca-pyg \
    && tar -xf /tmp/maca-pyg.tar.xz -C /tmp/maca-pyg --strip-components=1 \
    && python -m pip install /tmp/maca-pyg/wheel/*.whl \
    && rm -rf /tmp/maca-pyg /tmp/maca-pyg.tar.xz

RUN python -m pip install \
    e3nn==0.5.1 \
    torchmetrics==0.11.0 \
    fair-esm==2.0.0 \
    gradio==3.50.*

COPY ${DIFFDOCK_SOURCE} /opt/DiffDock

RUN cd /opt/DiffDock \
    && python utils/precompute_series.py

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'mkdir -p /opt/DiffDock/workdir/v1.1 "${TORCH_HOME}/hub/checkpoints"' \
    'if [ -d "${DIFFDOCK_WEIGHTS}" ]; then' \
    '  if [ -d "${DIFFDOCK_WEIGHTS}/score_model" ]; then' \
    '    ln -sfn "${DIFFDOCK_WEIGHTS}/score_model" /opt/DiffDock/workdir/v1.1/score_model' \
    '  fi' \
    '  if [ -d "${DIFFDOCK_WEIGHTS}/confidence_model" ]; then' \
    '    ln -sfn "${DIFFDOCK_WEIGHTS}/confidence_model" /opt/DiffDock/workdir/v1.1/confidence_model' \
    '  fi' \
    '  for file in esm2_t33_650M_UR50D.pt esm2_t33_650M_UR50D-contact-regression.pt; do' \
    '    if [ -f "${DIFFDOCK_WEIGHTS}/${file}" ]; then' \
    '      ln -sfn "${DIFFDOCK_WEIGHTS}/${file}" "${TORCH_HOME}/hub/checkpoints/${file}"' \
    '    fi' \
    '  done' \
    'fi' \
    'exec "$@"' \
    > /usr/local/bin/diffdock-entrypoint \
    && chmod +x /usr/local/bin/diffdock-entrypoint

WORKDIR /opt/DiffDock

ENTRYPOINT ["/usr/local/bin/diffdock-entrypoint"]
CMD ["/bin/bash"]
