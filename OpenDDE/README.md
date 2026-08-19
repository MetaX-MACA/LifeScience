# 沐曦 GPU 运行 OpenDDE 说明文档

[OpenDDE](https://github.com/aurekaresearch/OpenDDE) 项目由第三方以 Apache-2.0 许可证发布。本文档说明如何在沐曦曦云 C500 系列 GPU 上使用 MACA PyTorch 运行 OpenDDE-preview。

## 一、OpenDDE 简介

OpenDDE 是全原子生物分子基础模型，用于共折叠、结构预测、设计和药物发现相关任务。它通过 `opendde` CLI 提供 JSON 转换、MSA/模板预处理和预测。

## 二、已验证版本

本文档按以下本地源码版本整理，并已在 MACA PyTorch 基础镜像中完成 Docker 构建与最小扩散推理验证：

| 组件 | GitHub | 已核对 commit/tag |
| --- | --- | --- |
| OpenDDE | `https://github.com/aurekaresearch/OpenDDE.git` | `d42760d26463` (`v1.0.3`) |

本地权重目录中已核对的检查点文件名包括 `opendde.pt` 和 `opendde_abag.pt`。

已验证运行环境：

- 基础镜像：`cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64`
- PyTorch：`2.10.0+metax3.8.0.7`
- OpenDDE 包：`1.0.3`
- 最小推理：按第六节加载本地 `opendde.pt`，对 9 个氨基酸的仅蛋白质 JSON 运行 `opendde pred`，并确认输出 CIF、置信度摘要 JSON 和完整数据 JSON

## 三、环境准备

```bash
docker run -it --name test-opendde \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=10G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/workspace:/workspace \
  -v /path/to/opendde-data:/opendde_data:ro \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64 \
  /bin/bash
```

进入容器后：

```bash
export OPENDDE_ROOT_DIR=/opendde_data
export LAYERNORM_TYPE=torch
```

## 四、安装源码

从 GitHub 克隆源码并固定版本：

```bash
cd /workspace
git clone https://github.com/aurekaresearch/OpenDDE.git
cd OpenDDE
git checkout d42760d26463
```

OpenDDE 的 `pyproject.toml` 默认固定 `torch==2.7.1`，并把 `cuequivariance` 放在 `[gpu]` 额外依赖组。MACA 环境中不要安装 `[gpu]`，也不要覆盖镜像内 PyTorch：

```bash
python -m pip install --upgrade pip 'setuptools==84.0.0' wheel
sed -i '/"torch==/d; /gpu = \[/,/]/d' pyproject.toml
python -m pip install -e .
```

## 五、运行时数据与权重

OpenDDE 通过 `OPENDDE_ROOT_DIR` 查找检查点和运行时资源。建议宿主机目录组织为：

```text
/path/to/opendde-data/
├── checkpoint/
│   ├── opendde.pt
│   └── opendde_abag.pt
├── common/
└── search_database/
```

如果只做仅蛋白质兼容性预测，并关闭 MSA/模板/RNA-MSA，`search_database/` 不是必需的。

## 六、推理示例

创建最小输入：

```bash
cat > /workspace/tiny.json <<'EOF'
[
  {
    "name": "tiny",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIK",
          "count": 1
        }
      }
    ]
  }
]
EOF
```

运行兼容性预测，强制使用 PyTorch 三角计算内核：

```bash
opendde pred \
  -i /workspace/tiny.json \
  -o /workspace/opendde_output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --triatt_kernel torch \
  --trimul_kernel torch
```

使用 ABAG 检查点：

```bash
opendde pred \
  -i /workspace/tiny.json \
  -o /workspace/opendde_abag_output \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt" \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --triatt_kernel torch \
  --trimul_kernel torch
```

输出位于：

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```

## 七、cuequivariance 限制

当前 MACA PyTorch 镜像不支持 NVIDIA `cuequvariance`/`cuequivariance`。OpenDDE 的绕过方式是：

- 安装时不要使用 `opendde[gpu]`。
- 从 `pyproject.toml` 删除 `torch==...`，避免覆盖 MACA PyTorch。
- 运行时显式传入 `--triatt_kernel torch --trimul_kernel torch`。
- 多 GPU Fold-CP 也必须使用 PyTorch 三角计算内核。

## 八、Dockerfile 使用

准备源码：

```bash
cd /path/to/project-root
mkdir -p third_party
git clone https://github.com/aurekaresearch/OpenDDE.git third_party/OpenDDE
git -C third_party/OpenDDE checkout d42760d26463
```

构建并运行：

```bash
docker build \
  -f LifeScience/OpenDDE/Dockerfile \
  --build-arg OPENDDE_SOURCE=third_party/OpenDDE \
  -t opendde:maca-torch2.10 \
  .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=10G \
  -v /path/to/opendde-data:/opendde_data:ro \
  -v /path/to/workspace:/workspace \
  opendde:maca-torch2.10
```

## 九、维护与支持

沐曦仅维护本文档中的 MACA 环境配置说明。OpenDDE 的输入 JSON、预处理、检查点清单和内核选项请参考：

- [OpenDDE GitHub](https://github.com/aurekaresearch/OpenDDE)
- [OpenDDE 推理说明](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md)

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码；必要的本地安装修改已在文中列出。您按照本文档配置、部署或使用相关软件时，应遵守使用许可证的条款及条件。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
