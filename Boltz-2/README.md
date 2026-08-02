# 沐曦 GPU 运行 Boltz-2 说明文档

[Boltz-2](https://github.com/jwohlwend/boltz)项目由第三方以开源许可证发布，我方未对其源代码作任何修改。您可通过以下链接 https://github.com/jwohlwend/boltz 查阅本项目的源代码、许可证全文、版权与归属声明及其他声明文件。

## 一、Boltz-2 简介

[Boltz-2](https://github.com/jwohlwend/boltz) 是用于生物分子相互作用预测的开源模型，可用于蛋白质、核酸、小分子等复合物的结构预测，并支持结合亲和力相关预测。Boltz-2 在 Boltz-1 的结构预测基础上进一步联合建模 complex structure 与 binding affinity，适用于药物发现早期的 in silico screening、hit-to-lead 和 lead optimization 等场景。

本文档介绍如何在沐曦曦云 C500 系列 GPU 上使用 MACA PyTorch 2.8 环境部署并运行 Boltz-2。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 MACA PyTorch 镜像启动容器。本文档统一使用包含 PyTorch 2.8、Python 3.12 和 Ubuntu 24.04 的镜像：

```bash
docker run -it --name test-boltz-2 \
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
- `--shm-size=32G`：设置 shared memory，较大的 complex 或 batch inference 可继续调大。
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

### 2.3 下载 Boltz-2

```bash
apt-get update
apt-get install -y git wget

mkdir -p /workspace && cd /workspace
git clone https://github.com/jwohlwend/boltz.git
cd boltz
```

### 2.4 安装 Boltz-2

MACA PyTorch 已由基础镜像提供，不要安装上游 README 中面向 NVIDIA CUDA 的 `boltz[cuda]` extra，否则会尝试安装 `cuequivariance` 相关 CUDA package 并覆盖当前适配环境。

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## 三、Inference 示例

> 注意：当前 MACA PyTorch 镜像不支持 NVIDIA `cuequvariance` 。运行 Boltz-2 时需要使用 `--no-kernels` 关闭 kernels。

### 3.1 蛋白质结构预测

以下示例使用上游仓库自带的 `examples/prot.yaml`，并通过 MSA server 自动生成 MSA：

```bash
cd /workspace/boltz

boltz predict examples/prot.yaml \
  --use_msa_server \
  --out_dir results/prot \
  --no_kernels
```

### 3.2 结合亲和力预测

以下示例使用上游仓库自带的 `examples/affinity.yaml`：

```bash
cd /workspace/boltz

boltz predict examples/affinity.yaml \
  --use_msa_server \
  --out_dir results/affinity \
  --no_kernels
```

输出目录中会包含预测结构、置信度指标以及 affinity 相关结果。首次运行时，Boltz 会下载模型权重并缓存到默认 cache 目录，需保证网络连接和磁盘空间充足。

## 四、输入文件格式

Boltz-2 的输入可以是单个 YAML/FASTA 文件，也可以是包含多个输入文件的目录。YAML 文件通常包含 `sequences` 和可选的 `properties` 字段。

蛋白质结构预测示例：

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG
```

亲和力预测示例：

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
  - ligand:
      id: B
      smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
properties:
  - affinity:
      binder: B
```

更多输入格式请参考上游文档 `docs/prediction.md`。

## 五、常见问题与注意事项

### 5.1 不支持 cuequvariance/cuEquivariance kernels

本文档使用的 MACA PyTorch 镜像不支持 NVIDIA `cuequvariance`/`cuequivariance` kernels。请注意：

- 不要执行 `pip install "boltz[cuda]"`。
- 不要单独安装 `cuequivariance_ops_cu12`、`cuequivariance_ops_torch_cu12` 或 `cuequivariance_torch`。
- 运行 Boltz-2 时必须关闭 kernels，即使用 `--no_kernels`。

### 5.2 不要覆盖 MACA PyTorch

安装依赖时不要执行通用 CUDA 环境中的 `pip install torch` 或 `pip install torch --index-url https://download.pytorch.org/...`，否则可能覆盖镜像内的沐曦适配版 PyTorch。

### 5.3 MSA server 与模型权重下载

`--use_msa_server` 和首次模型权重下载都需要外网连接。若部署环境无法访问外网，请提前准备 MSA 文件和模型 cache，并通过 Boltz CLI 参数指定对应路径。

### 5.4 显存与运行时间

较长序列、多链 complex、较多 diffusion samples 或 affinity prediction 会增加显存占用。遇到 out of memory 时，可优先减少 `--diffusion_samples`、`--max_parallel_samples` 或输入规模。

## 六、Dockerfile 使用

本目录提供了可直接构建的 `Dockerfile`。镜像构建时从 build context 复制本地 `Boltz-2/boltz` 源码并安装，不把权重打入镜像；运行容器时将本机 Boltz-2 权重目录挂载到 `/weights/boltz2`。

```bash
cd /path/to/project-root

docker build \
  -f LifeScience/Boltz-2/Dockerfile \
  --build-arg BOLTZ_SOURCE=Boltz-2/boltz \
  -t boltz-2:maca-torch2.8 \
  .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=32G \
  -v /path/to/boltz2-weights:/weights/boltz2:ro \
  -v /path/to/workspace:/workspace \
  boltz-2:maca-torch2.8
```

容器启动脚本会把 `/weights/boltz2` 下的 `boltz2_conf.ckpt`、`boltz2_aff.ckpt` 和 `mols.tar` 链接到容器内 `BOLTZ_CACHE`。由于 host 目录只读挂载，`mols.tar` 首次解压会写入容器内 `/workspace/boltz-cache/mols`。

运行时仍需添加关闭 kernels 参数。示例：

```bash
boltz predict examples/prot.yaml \
  --use_msa_server \
  --out_dir /workspace/results/boltz2_prot \
  --no_kernels
```

## 七、Citation

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}
```

## 八、维护与支持

沐曦仅维护本文档中的 MACA environment 配置说明。Boltz-2 的 model architecture、training、prediction options 与权重发布细节请参考：

- [Boltz 官方 GitHub](https://github.com/jwohlwend/boltz)
- [Boltz-2 Technical Report](https://doi.org/10.1101/2025.06.14.659707)
- [沐曦开发者论坛](https://developer.metax-tech.com/forum/)

---

本文档仅提供相关 software 的配置与使用说明，不包含亦不分发前述 software 的 source code 或 object code，且不涉及对其 source code 的任何修改。您按照本文档配置、部署或使用相关 software 时，应遵守适用 license 的条款及条件。相关 software 的 source code、license 全文、copyright 与 attribution 声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.