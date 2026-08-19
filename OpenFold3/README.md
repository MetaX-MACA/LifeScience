# 沐曦 GPU 运行 OpenFold3 说明文档

[OpenFold3](https://github.com/aqlaboratory/openfold-3) 项目由第三方以 Apache-2.0 许可证发布。本文档说明如何在沐曦曦云 C500 系列 GPU 上使用 MACA PyTorch 运行 OpenFold3-preview 推理。

## 一、OpenFold3 简介

OpenFold3-preview 是 OpenFold 联盟发布的全原子生物分子结构预测模型，目标是复现 AlphaFold3 的输入特征与预测能力。它支持蛋白质、DNA、RNA、配体等多类分子，也支持 ColabFold MSA 服务、预计算 MSA 和免 MSA 推理。

## 二、已验证版本

本文档按以下本地源码版本整理，并已在 MACA PyTorch 基础镜像中完成 Docker 构建与最小推理验证：

| 组件 | GitHub | 已核对 commit |
| --- | --- | --- |
| OpenFold3 | `https://github.com/aqlaboratory/openfold-3.git` | `fb027a4e4f7e` |

本地权重目录中已核对的检查点文件名包括 `of3-p2-155k.pt`。

已验证运行环境：

- 基础镜像：`cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64`
- PyTorch：`2.10.0+metax3.8.0.7`
- OpenFold3 包：`0.4.3.dev1+gfb027a4e4.d20260814`
- 最小推理：按第七节免 MSA 示例加载 `of3-p2-155k.pt`，运行 `run_openfold predict`，并确认输出预测文件

## 三、环境准备

```bash
docker run -it --name test-openfold3 \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=10G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/workspace:/workspace \
  -v /path/to/openfold3-weights:/weights/openfold3:ro \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64 \
  /bin/bash
```

验证 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("gpu available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
```

## 四、安装源码

推荐从 GitHub 克隆源码并固定到已验证 commit：

```bash
cd /workspace
git clone https://github.com/aqlaboratory/openfold-3.git
cd openfold-3
git checkout fb027a4e4f7e
```

安装时保留镜像内 MACA PyTorch，并移除 NVIDIA `cuequivariance` 额外依赖组：

```bash
python -m pip install --upgrade pip 'setuptools==84.0.0' wheel
sed -i '/optional-dependencies.cuequivariance/,/]/d; /"torch"/d; /"torch>=/d' pyproject.toml
python -m pip install -e .
```

## 五、权重与本地数据

建议宿主机权重目录组织如下：

```text
/path/to/openfold3-weights/
└── models/
    └── of3-p2-155k.pt
```

README 示例在运行命令中显式指定检查点，不需要执行 `setup_openfold`，也不需要创建 `ckpt_root`：

```bash
--inference-ckpt-path /weights/openfold3/models/of3-p2-155k.pt
```

如果不传 `--inference-ckpt-path`，OpenFold3 会按上游默认逻辑从 `$OPENFOLD_CACHE` 或 `~/.openfold3` 查找/下载检查点。此时缓存目录必须可写，但不应指向只读的权重挂载目录。

OpenFold3 默认使用 Biotite 的 CCD 数据。当前验证镜像中 Biotite 已包含 `components.bcif`，仅蛋白质最小推理不需要额外挂载该文件。如需使用指定版本的 CCD，可在 runner YAML 中设置：

```yaml
dataset_config_kwargs:
  ccd_file_path: /path/to/components.bcif
```

## 六、关闭 cuequivariance/Triton/DeepSpeed 内核

当前 MACA PyTorch 镜像不支持 NVIDIA `cuequvariance`/`cuequivariance`。OpenFold3 上游还可能默认启用 Triton 或 DeepSpeed 内核。建议创建 runner YAML，强制使用 PyTorch 路径：

```bash
cat > /workspace/openfold3_no_cueq.yml <<'EOF'
model_update:
  presets:
    - predict
    - low_mem
  custom:
    settings:
      memory:
        eval:
          use_cueq_triangle_kernels: false
          use_deepspeed_evo_attention: false
          use_triton_triangle_kernels: false
EOF
```

不要安装或启用：

- `pip install openfold3[cuequivariance]`
- `cuequivariance`
- `cuequivariance-ops-torch-cu12`
- `cuequivariance-torch`

## 七、推理示例

### 7.1 使用 ColabFold MSA 服务

```bash
cd /workspace/openfold-3

run_openfold predict \
  --query-json examples/example_inference_inputs/query_ubiquitin.json \
  --use-msa-server \
  --output-dir /workspace/openfold3_outputs/ubiquitin \
  --runner-yaml /workspace/openfold3_no_cueq.yml \
  --inference-ckpt-path /weights/openfold3/models/of3-p2-155k.pt \
  --num-diffusion-samples 1
```

该模式需要访问 ColabFold MSA 服务。只会提交蛋白质序列用于 MSA 生成。

### 7.2 免 MSA 最小运行

```bash
cd /workspace/openfold-3

run_openfold predict \
  --query-json examples/example_inference_inputs/query_ubiquitin.json \
  --use-msa-server false \
  --use-templates false \
  --output-dir /workspace/openfold3_outputs/ubiquitin_no_msa \
  --runner-yaml /workspace/openfold3_no_cueq.yml \
  --inference-ckpt-path /weights/openfold3/models/of3-p2-155k.pt \
  --num-diffusion-samples 1
```

免 MSA 模式对疑难目标的预测质量可能下降，但更适合先验证环境、检查点和内核回退。

## 八、Dockerfile 使用

准备源码：

```bash
cd /path/to/project-root
mkdir -p third_party
git clone https://github.com/aqlaboratory/openfold-3.git third_party/openfold-3
git -C third_party/openfold-3 checkout fb027a4e4f7e
```

构建并运行：

```bash
docker build \
  -f LifeScience/OpenFold3/Dockerfile \
  --build-arg OPENFOLD3_SOURCE=third_party/openfold-3 \
  -t openfold3:maca-torch2.10 \
  .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=10G \
  -v /path/to/openfold3-weights:/weights/openfold3:ro \
  -v /path/to/workspace:/workspace \
  openfold3:maca-torch2.10
```

Dockerfile 内置 `/opt/openfold-3/maca/runner_no_cueq.yml`，运行时可以直接传给 `--runner-yaml`。

## 九、常见问题

- **安装时覆盖 torch**：不要直接执行上游 CUDA/ROCm 环境安装命令；先 `sed -i` 去掉 `torch` 依赖，再安装源码。
- **内核报错**：确认没有安装 `cuequivariance`，并且 runner YAML 中三个内核开关均为 `false`。
- **找不到检查点**：优先使用 `--inference-ckpt-path` 显式传入 `.pt` 文件。
- **OOM**：降低 `--num-diffusion-samples`，保留 `low_mem` 预设，并增加 `--shm-size`。

## 十、维护与支持

沐曦仅维护本文档中的 MACA 环境配置说明。OpenFold3 的模型架构、输入 JSON、MSA 流水线和权重发布细节请参考：

- [OpenFold3 GitHub](https://github.com/aqlaboratory/openfold-3)
- [OpenFold3 Documentation](https://openfold-3.readthedocs.io/)

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码；必要的本地安装修改已在文中列出。您按照本文档配置、部署或使用相关软件时，应遵守使用许可证的条款及条件。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
