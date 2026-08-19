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
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

ARG ESM_SOURCE=third_party/esm
ARG TRANSFORMERS_SOURCE=third_party/transformers
ARG DOCKQ_SOURCE=third_party/DockQ

FROM cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64

ARG ESM_SOURCE
ARG TRANSFORMERS_SOURCE
ARG DOCKQ_SOURCE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m pip install --upgrade pip 'setuptools==84.0.0' wheel

COPY ${TRANSFORMERS_SOURCE} /opt/transformers
COPY ${DOCKQ_SOURCE} /opt/DockQ
COPY ${ESM_SOURCE} /opt/esm

RUN python -c 'import importlib.metadata as md; [print(f"{p}=={md.version(p)}") for p in ("torch", "flash-attn")]' \
    > /tmp/maca-constraints.txt

RUN python -m pip install -c /tmp/maca-constraints.txt -e /opt/transformers \
    && python -m pip install -c /tmp/maca-constraints.txt -e /opt/DockQ \
    && sed -i '/"torch>=/d; /"transformers @ git+/d; /"dockq @ git+/d' /opt/esm/pyproject.toml \
    && python -m pip install -c /tmp/maca-constraints.txt -e /opt/esm

WORKDIR /opt/esm
CMD ["/bin/bash"]
