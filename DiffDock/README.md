# 沐曦 GPU 运行 DiffDock 说明文档

[DiffDock](https://github.com/gcorso/DiffDock)项目由第三方以开源许可证发布，我方未对其源代码作任何修改。您可通过以下链接 https://github.com/gcorso/DiffDock 查阅本项目的源代码、许可证全文、版权与归属声明及其他声明文件。

## 一、DiffDock 简介

[DiffDock](https://github.com/gcorso/DiffDock) 是一个基于 diffusion model 的 molecular docking 工具，用于预测小分子 ligand 与 protein receptor 的结合构象。当前上游仓库默认运行 DiffDock-L，该版本提升了 docking generalization 能力，适用于 small-molecule protein docking、候选 pose generation 和 docking confidence ranking 等任务。

本文档介绍如何在沐曦曦云 C500 系列 GPU 上使用 MACA PyTorch 2.8 环境部署并运行 DiffDock。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 MACA PyTorch 镜像启动容器。本文档统一使用包含 PyTorch 2.8、Python 3.12 和 Ubuntu 24.04 的镜像：

```bash
docker run -it --name test-diffdock \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=32G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/workspace:/workspace \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.8-py312-ubuntu24.04-amd64 \
  /bin/bash
```

参数说明：

- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦 GPU 设备。
- `--group-add video`：添加 `video` group，以访问 GPU。
- `--shm-size=32G`：设置 shared memory，batch inference 或 ESMFold 相关任务可继续调大。
- `-v /path/to/workspace:/workspace`：将 host 工作目录挂载到 container，可按实际路径修改。
- `maca-pytorch:3.8.0.11-torch2.8-py312-ubuntu24.04-amd64`：包含 MACA PyTorch 2.8 的基础镜像。

> 镜像版本需要与 host 上的 MACA Driver 兼容。其他系统或更新版本镜像请参考[沐曦开发者镜像资源中心](https://developer.metax-tech.com/softnova/docker/)。

### 2.2 验证 PyTorch 环境

进入 container 后执行：

```bash
python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

`PyTorch version` 应显示为 `2.8.x`，并且 `GPU available` 应为 `True`。

### 2.3 下载 DiffDock

```bash
apt-get update
apt-get install -y git wget unzip build-essential libmetis-dev

mkdir -p /workspace && cd /workspace
git clone https://github.com/gcorso/DiffDock.git
cd DiffDock
```

### 2.4 安装依赖

上游 `environment.yml` 和 `requirements.txt` 面向 NVIDIA CUDA 11.7 与 PyTorch 1.13.1，不能直接用于本文档的 MACA PyTorch 2.8 镜像。直接执行上游安装命令会覆盖镜像内的沐曦适配版 PyTorch。

建议按以下方式安装 Python dependencies，并保留基础镜像中的 `torch`。PyG 相关 wheel 使用开发者社区提供的 MACA PyG 包，例如：

```text
/path/to/maca-pyg-2.7.0-torch2.8-py312-3.8.0.11-ubuntu24.04-amd64.tar.xz
```

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
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

mkdir -p /tmp/maca-pyg
tar -xf /path/to/maca-pyg-2.7.0-torch2.8-py312-3.8.0.11-ubuntu24.04-amd64.tar.xz \
  -C /tmp/maca-pyg \
  --strip-components=1
python -m pip install /tmp/maca-pyg/wheel/*.whl

python -m pip install \
  e3nn==0.5.1 \
  torchmetrics==0.11.0 \
  fair-esm==2.0.0 \
  gradio==3.50.*
```

安装后检查关键依赖：

```bash
python - <<'PY'
import torch
import torch_geometric
import torch_cluster
import rdkit

print("torch:", torch.__version__, torch.cuda.is_available())
print("torch_geometric:", torch_geometric.__version__)
print("torch_cluster:", torch_cluster.__version__)
print("rdkit:", rdkit.__version__)
PY
```

> 如果 `torch-scatter`、`torch-sparse`、`torch-cluster` 或 `torch-spline-conv` 在当前网络环境下无法安装，请使用与 MACA PyTorch 2.8/Python 3.12 兼容的 wheel 或从源码构建。不要安装上游文档中的 `+cu117` wheels。

## 三、Inference 示例

### 3.1 使用单个 protein/ligand 文件

以下示例使用上游仓库自带的 sample PDB 和 SDF 文件：

```bash
cd /workspace/DiffDock

python -m inference \
  --config default_inference_args.yaml \
  --complex_name 1a46 \
  --protein_path examples/1a46_protein_processed.pdb \
  --ligand_description examples/1a46_ligand.sdf \
  --out_dir results/metax_1a46 \
  --samples_per_complex 10
```

首次运行时，DiffDock 会自动下载 model weights 到 `workdir/v1.1`。若环境无法访问外网，请提前下载上游 release 中的 `diffdock_models.zip` 并解压到 `workdir/v1.1`，使 `default_inference_args.yaml` 中的 `model_dir` 和 `confidence_model_dir` 指向有效目录。

### 3.2 使用 CSV 批量预测

CSV 需要包含 `complex_name`、`protein_path`、`ligand_description` 和 `protein_sequence` 四列。示例：

```bash
cd /workspace/DiffDock

cat > data/metax_inference.csv <<'EOF'
complex_name,protein_path,ligand_description,protein_sequence
1a46,examples/1a46_protein_processed.pdb,examples/1a46_ligand.sdf,
1a46_aspirin,examples/1a46_protein_processed.pdb,CC(=O)Oc1ccccc1C(=O)O,
EOF

python -m inference \
  --config default_inference_args.yaml \
  --protein_ligand_csv data/metax_inference.csv \
  --out_dir results/metax_batch \
  --samples_per_complex 10
```

输出结果保存在 `--out_dir` 指定目录下，每个 complex 会生成独立子目录，并包含按 confidence 排序的 ligand pose 文件。

## 四、常见问题与注意事项

### 4.1 不支持 cuequvariance/cuEquivariance

本文档使用的 MACA PyTorch 镜像不支持 NVIDIA `cuequvariance`/`cuequivariance` kernels。DiffDock 本身不需要安装 `cuequivariance`；如果其他 dependency 或本地脚本尝试安装相关 CUDA package，请跳过该部分。

### 4.2 不要覆盖 MACA PyTorch

不要直接执行以下上游 CUDA 安装方式：

```bash
conda env create --file environment.yml
python -m pip install -r requirements.txt
```

这些文件会安装 `torch==1.13.1+cu117` 以及 `torch-*-pt113cu117` packages，不适用于 `maca-pytorch:3.8.0.11-torch2.8-py312-ubuntu24.04-amd64`。

### 4.3 优先使用 PDB 输入

DiffDock 支持通过 `--protein_sequence` 调用 ESMFold 生成 protein structure，但 ESMFold/OpenFold 相关依赖更重，也更容易受 Python 与 PyTorch 版本影响。在 MACA PyTorch 2.8 镜像中，建议优先提供已准备好的 `.pdb` 文件。

### 4.4 PyG 扩展依赖

DiffDock 依赖 `torch_geometric` 以及 `torch_cluster` 等 PyG extension packages。此类 package 需要与当前 PyTorch 和 Python ABI 匹配。出现 `ModuleNotFoundError`、`undefined symbol` 或 device kernel 相关错误时，应检查是否误装了 CUDA 11.7/PyTorch 1.13 对应 wheels。

### 4.5 模型权重下载

DiffDock 默认从上游 GitHub release 下载模型权重。首次运行需要稳定网络连接和足够磁盘空间；离线环境请提前准备 `workdir/v1.1/score_model` 与 `workdir/v1.1/confidence_model`。

## 五、Dockerfile 使用

本目录提供了可直接构建的 `Dockerfile`。Dockerfile 会从 build context 复制本地 `DiffDock/DiffDock` 源码，并 `COPY` 已下载的 MACA PyG 压缩包；构建时请使用包含 `LifeScience/`、`DiffDock/DiffDock` 和 PyG 压缩包的项目根目录作为 build context：

```bash
cd /path/to/project-root

docker build \
  -f LifeScience/DiffDock/Dockerfile \
  --build-arg DIFFDOCK_SOURCE=DiffDock/DiffDock \
  --build-arg PYG_ARCHIVE=DiffDock/maca-pyg-2.7.0-torch2.8-py312-3.8.0.11-ubuntu24.04-amd64.tar.xz \
  -t diffdock:maca-torch2.8 \
  .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=32G \
  -v /path/to/diffdock-weights:/weights/diffdock:ro \
  -v /path/to/workspace:/workspace \
  diffdock:maca-torch2.8
```

容器启动脚本会把 `/weights/diffdock/score_model` 和 `/weights/diffdock/confidence_model` 链接到 `/opt/DiffDock/workdir/v1.1`，并把 ESM2 权重链接到 Torch Hub cache，避免首次推理重复下载。

## 六、Citation

```bibtex
@inproceedings{corso2023diffdock,
    title = {DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking},
    author = {Corso, Gabriele and Stark, Hannes and Jing, Bowen and Barzilay, Regina and Jaakkola, Tommi},
    booktitle = {International Conference on Learning Representations},
    year = {2023}
}
```

DiffDock-L 请同时引用：

```bibtex
@inproceedings{corso2024discovery,
    title = {Deep Confident Steps to New Pockets: Strategies for Docking Generalization},
    author = {Corso, Gabriele and Deng, Arthur and Polizzi, Nicholas and Barzilay, Regina and Jaakkola, Tommi},
    booktitle = {International Conference on Learning Representations},
    year = {2024}
}
```

## 七、维护与支持

沐曦仅维护本文档中的 MACA environment 配置说明。DiffDock 的 model architecture、training、evaluation 与 docking 细节请参考：

- [DiffDock 官方 GitHub](https://github.com/gcorso/DiffDock)
- [DiffDock Paper](https://arxiv.org/abs/2210.01776)
- [DiffDock-L Paper](https://arxiv.org/abs/2402.18396)
- [沐曦开发者论坛](https://developer.metax-tech.com/forum/)

---

本文档仅提供相关 software 的配置与使用说明，不包含亦不分发前述 software 的 source code 或 object code，且不涉及对其 source code 的任何修改。您按照本文档配置、部署或使用相关 software 时，应遵守适用 license 的条款及条件。相关 software 的 source code、license 全文、copyright 与 attribution 声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

