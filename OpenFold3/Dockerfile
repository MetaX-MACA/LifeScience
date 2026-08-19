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

ARG OPENFOLD3_SOURCE=third_party/openfold-3

FROM cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64

ARG OPENFOLD3_SOURCE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    hmmer \
    kalign \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m pip install --upgrade pip 'setuptools==84.0.0' wheel

COPY ${OPENFOLD3_SOURCE} /opt/openfold-3

RUN sed -i '/optional-dependencies.cuequivariance/,/]/d; /"torch"/d; /"torch>=/d' /opt/openfold-3/pyproject.toml \
    && python -m pip install -e /opt/openfold-3

RUN mkdir -p /opt/openfold-3/maca \
    && printf '%s\n' \
      'model_update:' \
      '  presets:' \
      '    - predict' \
      '    - low_mem' \
      '  custom:' \
      '    settings:' \
      '      memory:' \
      '        eval:' \
      '          use_cueq_triangle_kernels: false' \
      '          use_deepspeed_evo_attention: false' \
      '          use_triton_triangle_kernels: false' \
      > /opt/openfold-3/maca/runner_no_cueq.yml

WORKDIR /opt/openfold-3
CMD ["/bin/bash"]
