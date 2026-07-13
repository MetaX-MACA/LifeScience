# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
FROM cr.metax-tech.com/public-ai-release/maca/tensorflow2:2.13.1-maca.ai3.7.0.6-py310-ubuntu22.04-amd64
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda run -n base pip install --upgrade pip
RUN cd /opt && git clone --recursive https://github.com/nadavbra/protein_bert.git
RUN cd /opt/protein_bert && conda run -n base pip install .

