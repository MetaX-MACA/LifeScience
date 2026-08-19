# 沐曦 GPU 运行 BoltzGen 说明文档

[BoltzGen](https://github.com/HannesStark/boltzgen) 项目由第三方以 MIT 许可证发布。本文档说明如何在沐曦曦云 C500 系列 GPU 上使用 MACA PyTorch 运行 BoltzGen 蛋白质设计流水线。

## 一、BoltzGen 简介

BoltzGen 是面向结合蛋白设计的生成式模型。它以设计规格 YAML 为输入，生成排序后的设计结果，并可串联逆向折叠、折叠、亲和力预测、分析和过滤等步骤。

## 二、已验证版本

本文档按以下本地源码版本整理，并已在 MACA PyTorch 基础镜像中完成 Docker 构建与最小设计推理验证：

| 组件 | GitHub | 已核对 commit |
| --- | --- | --- |
| BoltzGen | `https://github.com/HannesStark/boltzgen.git` | `a3149cf18eeb` |

本地权重目录中已核对的文件包括 `boltzgen1_diverse.ckpt`、`boltzgen1_adherence.ckpt`、`boltzgen1_ifold.ckpt`、`boltz2_conf_final.ckpt`、`boltz2_aff.ckpt` 和 `mols.zip`。

已验证运行环境：

- 基础镜像：`cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64`
- PyTorch：`2.10.0+metax3.8.0.7`
- BoltzGen 包：`0.3.2`
- 最小推理：按第六节加载本地设计、逆向折叠、折叠和亲和力检查点，运行 `boltzgen run` 的 6 步流水线，并确认输出排序后的设计 CIF、指标 CSV 和摘要 PDF

## 三、环境准备

```bash
docker run -it --name test-boltzgen \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=10G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/workspace:/workspace \
  -v /path/to/boltzgen-weights:/weights/boltzgen:ro \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64 \
  /bin/bash
```

## 四、安装源码

从 GitHub 克隆源码并固定版本：

```bash
cd /workspace
git clone https://github.com/HannesStark/boltzgen.git
cd boltzgen
git checkout a3149cf18eeb
```

BoltzGen 上游把 `cuequivariance_ops_*` 放在普通依赖中。MACA 环境必须先移除这些依赖，并保留基础镜像内的 PyTorch：

```bash
python -m pip install --upgrade pip 'setuptools==84.0.0' wheel
sed -i '/"torch>=/d; /cuequivariance/d' pyproject.toml
python -m pip install -e .
```

## 五、权重与缓存

建议宿主机权重目录组织为：

```text
/path/to/boltzgen-weights/
├── boltzgen-1/
│   ├── boltzgen1_diverse.ckpt
│   ├── boltzgen1_adherence.ckpt
│   ├── boltzgen1_ifold.ckpt
│   ├── boltz2_conf_final.ckpt
│   └── boltz2_aff.ckpt
└── mols.zip
```

运行时可以显式传入所有检查点，避免首次运行从 Hugging Face 下载：

```bash
export BOLTZGEN_WEIGHTS=/weights/boltzgen/boltzgen-1
export BOLTZGEN_CACHE=/workspace/boltzgen-cache
mkdir -p "$BOLTZGEN_CACHE"
```

## 六、推理示例

先检查 YAML 规格：

```bash
cd /workspace/boltzgen

boltzgen check \
  example/hard_targets/1g13prot.yaml \
  --output /workspace/boltzgen_checked \
  --moldir /weights/boltzgen/mols.zip
```

运行一个小规模设计推理。核心绕过参数是 `--use_kernels false`：

```bash
boltzgen run example/hard_targets/1g13prot.yaml \
  --output /workspace/boltzgen_test_run \
  --protocol protein-anything \
  --num_designs 2 \
  --budget 1 \
  --use_kernels false \
  --cache "$BOLTZGEN_CACHE" \
  --design_checkpoints \
    "$BOLTZGEN_WEIGHTS/boltzgen1_diverse.ckpt" \
    "$BOLTZGEN_WEIGHTS/boltzgen1_adherence.ckpt" \
  --inverse_fold_checkpoint "$BOLTZGEN_WEIGHTS/boltzgen1_ifold.ckpt" \
  --folding_checkpoint "$BOLTZGEN_WEIGHTS/boltz2_conf_final.ckpt" \
  --affinity_checkpoint "$BOLTZGEN_WEIGHTS/boltz2_aff.ckpt" \
  --moldir /weights/boltzgen/mols.zip
```

常规生产运行需要显著增加 `--num_designs`。先用小规模运行确认输入 YAML、检查点、缓存和内核回退正常，再扩展到更大设计数。

## 七、cuequivariance 限制

当前 MACA PyTorch 镜像不支持 NVIDIA `cuequvariance`/`cuequivariance`。BoltzGen 的绕过方式是：

- 安装前用 `sed -i '/cuequivariance/d' pyproject.toml` 删除相关依赖。
- 安装前删除 `torch>=...`，避免覆盖 MACA PyTorch。
- 运行 `boltzgen run` 时传入 `--use_kernels false`。
- 不安装 `cuequivariance_ops_cu12`、`cuequivariance_ops_torch_cu12`、`cuequivariance_torch`。

## 八、Dockerfile 使用

准备源码：

```bash
cd /path/to/project-root
mkdir -p third_party
git clone https://github.com/HannesStark/boltzgen.git third_party/boltzgen
git -C third_party/boltzgen checkout a3149cf18eeb
```

构建并运行：

```bash
docker build \
  -f LifeScience/BoltzGen/Dockerfile \
  --build-arg BOLTZGEN_SOURCE=third_party/boltzgen \
  -t boltzgen:maca-torch2.10 \
  .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=10G \
  -v /path/to/boltzgen-weights:/weights/boltzgen:ro \
  -v /path/to/workspace:/workspace \
  boltzgen:maca-torch2.10
```

## 九、维护与支持

沐曦仅维护本文档中的 MACA 环境配置说明。BoltzGen 的设计规格、协议、流水线步骤和完整 CLI 参数请参考：

- [BoltzGen GitHub](https://github.com/HannesStark/boltzgen)
- [BoltzGen paper](https://hannes-stark.com/assets/boltzgen.pdf)

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码；必要的本地安装修改已在文中列出。您按照本文档配置、部署或使用相关软件时，应遵守使用许可证的条款及条件。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
