# 沐曦 GPU 运行 BioT5+ 说明文档

## 一、BioT5+ 简介

[BioT5+](https://github.com/QizhiPei/BioT5/tree/main/biot5_plus) 是面向 biology 与 chemistry 的 pre-trained language model。该模型基于 T5 encoder-decoder architecture，在 natural language、protein sequence、molecule SELFIES 和 IUPAC name 等多种 biological modality 之间建立统一表示，并通过 multi-task instruction tuning 提升对不同任务的泛化能力。

BioT5+ 支持 Molecule Captioning、Text-based Molecule Generation、molecule understanding、reaction prediction、protein function understanding 和 protein design 等任务。本文档介绍如何在沐曦曦云 C500 系列 GPU 上使用 PyTorch 2.4 环境部署并运行 BioT5+。

## 二、沐曦 GPU 环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的 [MACA PyTorch 镜像](https://developer.metax-tech.com/softnova/docker/) 启动容器。本文档统一使用包含 PyTorch 2.4 和 Python 3.10 的镜像：

```bash
docker run -it --name test-biot5-plus \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=16G \
  -v /home/user:/workspace \
  cr.metax-tech.com/public-library/maca-c500-pytorch:3.8.0.11-ubuntu22.04-amd64 \
  /bin/bash
```

参数说明：

- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦 GPU 设备。
- `--group-add video`：添加 `video` group，以访问 GPU。
- `--shm-size=16G`：设置 shared memory，执行 evaluation 时可根据 batch size 调整。
- `-v /home/user:/workspace`：将 host 工作目录挂载到 container，可按实际路径修改。
- `maca-c500-pytorch:3.8.0.11-ubuntu22.04-amd64`：包含 PyTorch 2.4 的曦云 C500 基础镜像。

> 镜像版本需要与 host 上的 MACA Driver 兼容。其他系统或更新版本镜像请参考[沐曦开发者镜像资源中心](https://developer.metax-tech.com/softnova/docker/)。

### 2.2 验证 PyTorch 2.4 环境

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

`PyTorch version` 应显示为 `2.4.0+metax3.8.0.7`，并且 `GPU available` 应为 `True`。

### 2.3 下载 BioT5+

BioT5+ 的 fine-tuning data 通过 Git submodule 和 Git LFS 管理。先安装 Git LFS，再 recursive clone 上游仓库：

```bash
apt-get update
apt-get install -y git git-lfs
git lfs install

mkdir -p /workspace && cd /workspace
git clone --recursive https://github.com/QizhiPei/BioT5.git
cd BioT5/biot5_plus
```

如果已经 clone 仓库但未下载 submodule，可在 `BioT5` 根目录补充执行：

```bash
git submodule update --init --recursive
```

### 2.4 安装依赖

PyTorch 2.4 已由 MACA image 提供，无需执行上游文档中的 `pip install torch==2.0.1`，否则可能覆盖沐曦适配版 PyTorch。

```bash
cd /workspace/BioT5/biot5_plus
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装完成后可再次检查 PyTorch，确认版本仍为 `2.4.x`：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 三、Inference 示例

以下示例使用 fine-tuned checkpoint [`QizhiPei/biot5-plus-base-chebi20`](https://huggingface.co/QizhiPei/biot5-plus-base-chebi20)。首次运行时，Transformers 会自动从 Hugging Face 下载 model weights 和 tokenizer。

### 3.1 Molecule Captioning

根据 molecule SELFIES 生成 English molecule description：

```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

checkpoint = "QizhiPei/biot5-plus-base-chebi20"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(checkpoint, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
model.eval()

task_definition = (
    "Definition: You are given a molecule SELFIES. Your job is to generate "
    "the molecule description in English that fits the molecule SELFIES.\n\n"
)
selfies_input = "[C][C][Branch1][C][O][C][C][=Branch1][C][=O][C][=Branch1][C][=O][O-1]"
task_input = (
    "Now complete the following example -\n"
    f"Input: <bom>{selfies_input}<eom>\n"
    "Output: "
)

model_input = task_definition + task_input
input_ids = tokenizer(model_input, return_tensors="pt").input_ids.to(device)

generation_config = model.generation_config
generation_config.max_length = 512
generation_config.num_beams = 1

with torch.no_grad():
    outputs = model.generate(input_ids, generation_config=generation_config)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

结果为

```text
The molecule is a hydroxy monocarboxylic acid anion that is the conjugate base of 4 -hydroxy- 2 -oxopentanoic acid. It is a hydroxy monocarboxylic acid anion and an oxo fatty acid anion. It derives from a valerate. It is a conjugate base of a 4 -hydroxy- 2 -oxopentanoic acid.
```

### 3.2 Text-based Molecule Generation

根据 English molecule description 生成 SELFIES，并转换为 SMILES：

```python
import selfies as sf
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

checkpoint = "QizhiPei/biot5-plus-base-chebi20"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(checkpoint, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
model.eval()

task_definition = (
    "Definition: You are given a molecule description in English. Your job is "
    "to generate the molecule SELFIES that fits the description.\n\n"
)
text_input = (
    "The molecule is a monocarboxylic acid anion obtained by deprotonation of "
    "the carboxy and sulfino groups of 3-sulfinopropionic acid. Major "
    "microspecies at pH 7.3 It is an organosulfinate oxoanion and a "
    "monocarboxylic acid anion. It is a conjugate base of a "
    "3-sulfinopropionic acid."
)
task_input = (
    "Now complete the following example -\n"
    f"Input: {text_input}\n"
    "Output: "
)

model_input = task_definition + task_input
input_ids = tokenizer(model_input, return_tensors="pt").input_ids.to(device)

generation_config = model.generation_config
generation_config.max_length = 512
generation_config.num_beams = 1

with torch.no_grad():
    outputs = model.generate(input_ids, generation_config=generation_config)

output_selfies = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
output_smiles = sf.decoder(output_selfies)

print("SELFIES:", output_selfies)
print("SMILES:", output_smiles)
```

结果为

```text
SELFIES: [O][=C][Branch1][C][O-1][C][C][S][=Branch1][C][=O][O-1]
SMILES: O=C([O-1])CCS(=O)[O-1]
```

## 四、Data 与 Model Checkpoints

### 4.1 Fine-tuning Data

BioT5+ 的 instruction-formatted fine-tuning data 发布在 [`QizhiPei/BioT5_finetune_dataset`](https://huggingface.co/datasets/QizhiPei/BioT5_finetune_dataset)。使用 `--recursive` clone BioT5 时，data 会作为 submodule 下载到仓库根目录的 `data` 目录。

若未使用 recursive clone，可手动下载：

```bash
cd /workspace/BioT5
git clone https://huggingface.co/datasets/QizhiPei/BioT5_finetune_dataset data
```

### 4.2 Model Checkpoints

| Model | Description | Hugging Face Checkpoint |
| --- | --- | --- |
| BioT5-plus-base | Pre-trained BioT5+ base model | [QizhiPei/biot5-plus-base](https://huggingface.co/QizhiPei/biot5-plus-base) |
| BioT5-plus-large | Pre-trained BioT5+ large model | [QizhiPei/biot5-plus-large](https://huggingface.co/QizhiPei/biot5-plus-large) |
| BioT5-plus-mol-instructions (molecule) | 在 Mol-Instructions molecule tasks 上 fine-tuned | [QizhiPei/biot5-plus-base-mol-instructions-molecule](https://huggingface.co/QizhiPei/biot5-plus-base-mol-instructions-molecule) |
| BioT5-plus-mol-instructions (protein) | 在 Mol-Instructions protein tasks 上 fine-tuned | [QizhiPei/biot5-plus-base-mol-instructions-protein](https://huggingface.co/QizhiPei/biot5-plus-base-mol-instructions-protein) |
| BioT5-plus-chebi20 | 用于 Molecule Captioning 与 Text-based Molecule Generation | [QizhiPei/biot5-plus-base-chebi20](https://huggingface.co/QizhiPei/biot5-plus-base-chebi20) |

## 五、Evaluation

BioT5+ 提供 ChEBI-20 和 Mol-Instructions 的 evaluation scripts。

### 5.1 准备 Checkpoint

可以让 Transformers 在运行时自动下载 checkpoint，也可以提前下载到本地：

```bash
mkdir -p /workspace/models
git clone https://huggingface.co/QizhiPei/biot5-plus-base-chebi20 \
  /workspace/models/biot5-plus-base-chebi20
```

运行 evaluation 前，需要将对应 shell script 第一行的 `ckpt_path` 修改为 checkpoint 路径，例如：

```bash
ckpt_path=/workspace/models/biot5-plus-base-chebi20
```

### 5.2 执行 Evaluation

```bash
cd /workspace/BioT5/biot5_plus

# ChEBI-20: Molecule Captioning 与 Text-based Molecule Generation
bash evaluation_chebi20.sh

# Mol-Instructions: molecule-related tasks
bash evaluation_mol_instruction_molecule.sh

# Mol-Instructions: protein-related tasks
bash evaluation_mol_instruction_protein.sh
```

Evaluation 结果默认保存在 scripts 中 `run_dir` 指定的目录。

## 六、常见问题与注意事项

### 6.1 不要重新安装 PyTorch

本文档要求使用 `torch2.4` image。上游 README 中的 PyTorch 2.0.1 安装命令面向通用 CUDA 环境，不适用于此处的 MACA environment。安装 dependencies 时不要覆盖 image 中的沐曦适配版 PyTorch。

### 6.2 Hugging Face 下载

Model weights 与 fine-tuning data 文件较大，首次下载需要稳定的网络连接和足够的磁盘空间。若使用 Git clone 下载 Hugging Face checkpoint 或 dataset，需要提前执行 `git lfs install`。

### 6.3 GPU Memory

`BioT5-plus-large`、较长 sequence length 或较大的 batch size 会占用更多 GPU memory。遇到 out of memory 时，可优先减小 `optim.batch_size`、`data.max_seq_len` 或 generation 的 `max_length`。

### 6.4 Hydra 输出目录

Training 与 evaluation 入口 `main.py` 使用 Hydra 管理 configuration。运行目录由 `hydra.run.dir` 控制，logs、checkpoint 和 prediction results 会写入对应目录。可通过 command line override 修改，例如：

```bash
python main.py hydra.run.dir=/workspace/logs/biot5-plus-run
```

## 七、Dockerfile 示例

```Dockerfile
FROM cr.metax-tech.com/public-library/maca-c500-pytorch:3.8.0.11-ubuntu22.04-amd64

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m pip install --upgrade pip

RUN git clone --recursive https://github.com/QizhiPei/BioT5.git /opt/BioT5 \
    && python -m pip install -r /opt/BioT5/biot5_plus/requirements.txt

WORKDIR /opt/BioT5/biot5_plus

CMD ["/bin/bash"]
```

构建和启动：

```bash
docker build -t biot5-plus:torch2.4 .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=16G \
  biot5-plus:torch2.4
```

## 八、Citation

```bibtex
@article{pei2024biot5+,
  title={BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning},
  author={Pei, Qizhi and Wu, Lijun and Gao, Kaiyuan and Liang, Xiaozhuan and Fang, Yin and Zhu, Jinhua and Xie, Shufang and Qin, Tao and Yan, Rui},
  journal={arXiv preprint arXiv:2402.17810},
  year={2024}
}
```

## 九、维护与支持

沐曦仅维护本文档中的 MACA environment 配置说明。BioT5+ 的 model architecture、data、training 与 evaluation 细节请参考：

- [BioT5+ 官方 GitHub](https://github.com/QizhiPei/BioT5/tree/main/biot5_plus)
- [BioT5+ Paper](https://arxiv.org/abs/2402.17810)
- [BioT5+ Hugging Face Models](https://huggingface.co/QizhiPei)
- [沐曦开发者论坛](https://developer.metax-tech.com/forum/)

本文档仅提供相关 software 的配置与使用说明，不包含亦不分发前述 software 的 source code 或 object code，且不涉及对其 source code 的任何修改。您按照本文档配置、部署或使用相关 software 时，应遵守适用 license 的条款及条件。相关 software 的 source code、license 全文、copyright 与 attribution 声明及其他项目文档，请以其官方网站或原始发布页面为准。
