# 沐曦 GPU 运行 ESM3、ESMC 与 ESMFold2 说明文档

[ESM](https://github.com/Biohub/esm) 项目由第三方以开源许可证发布。本文档只说明在沐曦曦云 C500 系列 GPU 上的 MACA PyTorch 环境适配方式，不分发上游源码或模型权重。

## 一、模型简介

ESM3、ESMC 与 ESMFold2 是 Biohub 发布的蛋白质基础模型系列：

- **ESM3**：面向蛋白质序列、结构和功能的生成与补全模型。
- **ESMC**：蛋白质语言模型，可用于序列嵌入、掩码标记预测和隐藏状态分析。
- **ESMFold2**：基于 ESMC 6B 嵌入的全原子结构预测模型，支持蛋白质、DNA/RNA 与配体输入。

本文档覆盖本地源码安装、离线权重挂载和最小推理示例。

## 二、已验证版本

本文档按以下本地源码版本整理，并已在 MACA PyTorch 基础镜像中完成 Docker 构建、ESMC 300M 前向推理与 ESMFold2 最小折叠推理验证：

| 组件 | GitHub | 已核对 commit/tag |
| --- | --- | --- |
| ESM | `https://github.com/Biohub/esm.git` | `26b0bc2b771e` (`v3.2.3-24-g26b0bc2`) |
| Biohub Transformers fork | `https://github.com/Biohub/transformers.git` | `ef32577f55da` |
| DockQ fork | `https://github.com/nrontsis/DockQ.git` | `ba4df5adaad7` |

建议生产环境固定到上述 commit。上游 `esm` 当前通过 Git URL 依赖 Biohub Transformers 和 DockQ；在受限网络或离线环境中，应先克隆对应源码，再按本文档本地安装。

已验证运行环境：

- 基础镜像：`cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64`
- PyTorch：`2.10.0+metax3.8.0.7`
- flash-attn：`2.6.3+metax3.8.0.7torch2.10`

## 三、环境准备

启动 MACA PyTorch 容器：

```bash
docker run -it --name test-esm3 \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=64G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/workspace:/workspace \
  -v /path/to/esm3-weights:/weights/esm3:ro \
  -v /path/to/biohub-hf:/weights/biohub-hf:ro \
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
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY
```

## 四、安装源码

推荐从 GitHub 克隆源码，并固定到已验证 commit：

```bash
cd /workspace
mkdir -p third_party && cd third_party

git clone https://github.com/Biohub/transformers.git
git -C transformers checkout ef32577f55da

git clone https://github.com/nrontsis/DockQ.git
git -C DockQ checkout ba4df5adaad7

git clone https://github.com/Biohub/esm.git
git -C esm checkout 26b0bc2b771e
```

安装时保留镜像内的 MACA PyTorch 和 `flash_attn`，不让 `pip` 重新安装通用 `torch` 或替换镜像内置的 `flash-attn` wheel。
不要安装 Biohub Transformers 的 `torch`、`all` 或 test/dev 额外依赖。

## 五、权重组织

ESM3 的上游加载器会调用 `huggingface_hub.snapshot_download()`，离线环境可以把本地 ESM3 权重映射到 Hugging Face 缓存布局，从而保持上游 ESM 源码不变。

ESMC 300M/600M 的本地权重如果是上游发布压缩包中的 `.pth` 文件，建议按 6.1 的示例显式加载 `.pth`。当前已验证源码版本中的 `ESMC.from_pretrained()` 通过 `huggingface_hub.load_torch_model()` 读取检查点，不能直接加载 `data/weights/*.pth` 这种目录布局；没有必要为此修改 ESM 源码。

```text
/path/to/esm3-weights/
├── esm3-sm-open-v1/
├── esmc-300m-2024-12/
│   └── data/weights/esmc_300m_2024_12_v0.pth
└── esmc-600m-2024-12/
    └── data/weights/esmc_600m_2024_12_v0.pth
```

ESMC-6B 与 ESMFold2 使用 Hugging Face Transformers 格式。建议将镜像目录整理为：

```text
/path/to/biohub-hf/
├── ESMC-6B/
├── ESMFold2/
└── ESMFold2-Fast/
```

进入容器后设置：

```bash
export HF_HOME=/workspace/hf-cache
export HF_HUB_OFFLINE=1
```

然后为 ESM3 创建缓存符号链接。`refs/main` 的内容只要和 `snapshots/<revision>` 名称一致即可，本地离线使用时可统一写成 `local`：

```bash
mkdir -p "$HF_HOME/hub"

repo=esm3-sm-open-v1
cache_dir="$HF_HOME/hub/models--biohub--${repo}"
mkdir -p "$cache_dir/refs" "$cache_dir/snapshots"
printf "%s" "local" > "$cache_dir/refs/main"
ln -sfn "/weights/esm3/${repo}" "$cache_dir/snapshots/local"
```

ESMC-6B 与 ESMFold2 使用 Transformers `from_pretrained()`。ESMFold2 的构建过程内部还会用仓库 ID 解析 `biohub/ESMC-6B` 和 `biohub/ESMFold2` 中的 `ccd.pkl`，离线运行时建议把本地 Hugging Face 目录映射到缓存布局。当前镜像同时设置了 `HF_HOME` 和 `TRANSFORMERS_CACHE`，因此为兼容 Transformers/Hugging Face Hub 的不同查找路径，建议在 `$HF_HOME` 和 `$HF_HOME/hub` 下都创建符号链接：

```bash
for repo in ESMC-6B ESMFold2 ESMFold2-Fast; do
  if [ -d "/weights/biohub-hf/${repo}" ]; then
    for root in "$HF_HOME" "$HF_HOME/hub"; do
      cache_dir="$root/models--biohub--${repo}"
      mkdir -p "$cache_dir/refs" "$cache_dir/snapshots"
      printf "%s" "local" > "$cache_dir/refs/main"
      ln -sfn "/weights/biohub-hf/${repo}" "$cache_dir/snapshots/local"
    done
  fi
done
```

ESMFold2 示例仍显式从 `/weights/biohub-hf/ESMFold2` 加载主模型；上述缓存符号链接用于满足模型内部对 `biohub/ESMC-6B` 和 CCD 的离线查找。

## 六、推理示例

### 6.1 ESMC 嵌入

```bash
python - <<'PY'
from pathlib import Path

import torch
from accelerate import init_empty_weights
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = Path("/weights/esm3/esmc-300m-2024-12/data/weights/esmc_300m_2024_12_v0.pth")

with init_empty_weights():
    model = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=True,
    ).eval()

state = torch.load(weights, map_location="cpu")
model.load_state_dict(state, assign=True)
model = model.to(device)
if device.type != "cpu":
    model = model.to(torch.bfloat16)

tokens = model._tokenize(["MKTAYIAKQRQISFVKSHFSRQDILDLWQ"])

with torch.inference_mode():
    out = model(tokens)

print("sequence logits:", tuple(out.sequence_logits.shape))
print("embeddings:", None if out.embeddings is None else tuple(out.embeddings.shape))
PY
```

本地已验证 ESMC 300M 输出：

```text
sequence logits: (1, 31, 64)
embeddings: (1, 31, 960)
```

如果使用 ESMC 600M `.pth`，对应参数为 `d_model=1152`、`n_heads=18`、`n_layers=36`，权重路径改为 `esmc_600m_2024_12_v0.pth`。更大的 ESMC-6B 建议通过 `transformers.AutoModel` 从本地 Hugging Face 目录加载。

### 6.2 ESMFold2

```bash
python - <<'PY'
from pathlib import Path

from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

model_dir = Path("/weights/biohub-hf/ESMFold2")
model = ESMFold2Model.from_pretrained(model_dir).cuda().eval()

spi = StructurePredictionInput(
    sequences=[
        ProteinInput(
            id="A",
            sequence="MKTAYIAKQRQISFVKSHFSRQDILDLWQ",
        )
    ]
)

result = ESMFold2InputBuilder().fold(
    model,
    spi,
    num_loops=4,
    num_sampling_steps=20,
    num_diffusion_samples=1,
    seed=0,
)

Path("/workspace/esmfold2_test.cif").write_text(result.complex.to_mmcif())
print("pLDDT mean:", float(result.plddt.mean()))
PY
```

本地已验证 ESMFold2 输出示例：

```text
pLDDT mean: 0.6739968657493591
```

蛋白质结构保存在 `/workspace/esmfold2_test.cif` 中

## 七、Dockerfile 使用

本目录 Dockerfile 使用本地源码构建镜像。先从 GitHub 准备源码：

```bash
cd /path/to/project-root
mkdir -p third_party
git clone https://github.com/Biohub/esm.git third_party/esm
git clone https://github.com/Biohub/transformers.git third_party/transformers
git clone https://github.com/nrontsis/DockQ.git third_party/DockQ
git -C third_party/esm checkout 26b0bc2b771e
git -C third_party/transformers checkout ef32577f55da
git -C third_party/DockQ checkout ba4df5adaad7
```

构建并运行：

```bash
docker build \
  -f LifeScience/ESM3/Dockerfile \
  --build-arg ESM_SOURCE=third_party/esm \
  --build-arg TRANSFORMERS_SOURCE=third_party/transformers \
  --build-arg DOCKQ_SOURCE=third_party/DockQ \
  -t esm3:maca-torch2.10 \
  .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=64G \
  -v /path/to/esm3-weights:/weights/esm3:ro \
  -v /path/to/biohub-hf:/weights/biohub-hf:ro \
  -v /path/to/workspace:/workspace \
  esm3:maca-torch2.10
```

## 八、维护与支持

沐曦仅维护本文档中的 MACA 环境配置说明。ESM3、ESMC、ESMFold2 的模型架构、权重许可、输入格式和完整 API 请参考：

- [Biohub ESM GitHub](https://github.com/Biohub/esm)
- [Biohub ESMC 模型系列](https://huggingface.co/collections/biohub/esmc-model-family)
- [Biohub ESMFold2](https://huggingface.co/biohub/ESMFold2)

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码；必要的本地安装修改已在文中列出。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证的条款及条件。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
