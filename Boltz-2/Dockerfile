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

ARG BOLTZ_SOURCE=Boltz-2/boltz

FROM cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.8-py312-ubuntu24.04-amd64

ARG BOLTZ_SOURCE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BOLTZ_WEIGHTS=/weights/boltz2 \
    BOLTZ_CACHE=/workspace/boltz-cache

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY ${BOLTZ_SOURCE} /opt/boltz

RUN python -m pip install -e /opt/boltz

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'mkdir -p "${BOLTZ_CACHE}"' \
    'if [ -d "${BOLTZ_WEIGHTS}" ]; then' \
    '  for file in boltz2_conf.ckpt boltz2_aff.ckpt mols.tar ccd.pkl; do' \
    '    if [ -f "${BOLTZ_WEIGHTS}/${file}" ]; then' \
    '      ln -sfn "${BOLTZ_WEIGHTS}/${file}" "${BOLTZ_CACHE}/${file}"' \
    '    fi' \
    '  done' \
    '  if [ -d "${BOLTZ_WEIGHTS}/mols" ] && [ ! -e "${BOLTZ_CACHE}/mols" ]; then' \
    '    ln -s "${BOLTZ_WEIGHTS}/mols" "${BOLTZ_CACHE}/mols"' \
    '  fi' \
    'fi' \
    'exec "$@"' \
    > /usr/local/bin/boltz-entrypoint \
    && chmod +x /usr/local/bin/boltz-entrypoint

WORKDIR /opt/boltz

ENTRYPOINT ["/usr/local/bin/boltz-entrypoint"]
CMD ["/bin/bash"]
