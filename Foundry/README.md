# 沐曦 GPU 运行 RoseTTAFold3、RFdiffusion3、RFD3NA、ProteinMPNN 与 LigandMPNN 说明文档

[Foundry](https://github.com/RosettaCommons/foundry) 项目由 RosettaCommons 以 BSD-3-Clause 许可证发布。本文档说明如何在沐曦曦云 C500 系列 GPU 上使用 MACA PyTorch 运行 Foundry 中的 RF3、RFD3、RFD3NA、ProteinMPNN 和 LigandMPNN。

## 一、模型简介

Foundry 集成了多类蛋白质设计与结构预测模型：

- **RoseTTAFold3 / RF3**：全原子生物分子结构预测。
- **RFdiffusion3 / RFD3**：全原子生成式设计模型，可用于蛋白质互作结合蛋白、小分子结合蛋白、酶和对称结构设计。
- **RFD3NA**：RFD3 的核苷酸扩展，可设计蛋白质-DNA-RNA 等多聚体系统。
- **ProteinMPNN**：固定骨架的蛋白质序列设计。
- **LigandMPNN**：带配体/原子上下文的固定骨架序列设计。

## 二、已验证版本

本文档按以下本地源码版本整理，并已在 MACA PyTorch 基础镜像中完成 Docker 构建与最小推理验证：

| 组件 | GitHub | 已核对 commit |
| --- | --- | --- |
| Foundry | `https://github.com/RosettaCommons/foundry.git` | `4010e3e2e735` |

本地权重目录中已核对的文件包括 `rf3_foundry_01_24_latest_remapped.ckpt`、`rfd3_latest.ckpt`、`rfd3na-1190.ckpt`、`proteinmpnn_v_48_020.pt` 和 `ligandmpnn_v_32_010_25.pt`。

已验证运行环境：

- 基础镜像：`cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64`
- PyTorch：`2.10.0+metax3.8.0.7`
- rc-foundry 包：`0.0.1.dev1+g4010e3e2e.d20260814`
- atomworks 包：`2.2.1`
- 最小推理：按第六节加载本地检查点，完成 RF3 结构预测、RFD3 设计、RFD3NA 设计、ProteinMPNN 序列设计和 LigandMPNN 序列设计，并确认输出 CIF/JSON/FASTA 文件

## 三、环境准备

```bash
docker run -it --name test-foundry \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=64G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/workspace:/workspace \
  -v /path/to/foundry-weights:/weights/foundry:ro \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64 \
  /bin/bash
```

进入容器后：

```bash
export FOUNDRY_CHECKPOINT_DIRS=/weights/foundry
export DISABLE_CUEQUIVARIANCE=1
```

## 四、安装源码

从 GitHub 克隆源码并固定版本：

```bash
cd /workspace
git clone https://github.com/RosettaCommons/foundry.git
cd foundry
git checkout 4010e3e2e735
```

Foundry 的 `rf3`/`all` 额外依赖组会安装 `cuequivariance`。
MACA 环境还需要修正 RFD3/RFD3NA 中 apex RMSNorm 的 shape 类型：

```bash
python -m pip install --upgrade pip 'setuptools==84.0.0' wheel

sed -i '/"torch>=/d; /atomworks\[ml\]/s/atomworks\[ml\]/atomworks/; /cuequivariance/d; /rc-foundry\[rf3\]/d' pyproject.toml
sed -i 's/out_features = np.prod(out_shape)/out_features = int(np.prod(out_shape))/' \
  models/rfd3/src/rfd3/model/layers/layer_utils.py \
  models/rfd3na/src/rfd3na/model/layers/layer_utils.py
python -m pip install -e .
```

如果你的任务需要 `atomworks[ml]` 中的训练数据处理能力，请先确认该额外依赖组不会安装 `cuequivariance`、替换 `torch`；否则仍应手动拆分依赖，只安装推理所需部分。

## 五、权重组织

建议宿主机权重目录组织为：

```text
/path/to/foundry-weights/
├── rf3_foundry_01_24_latest_remapped.ckpt
├── rf3_foundry_01_24_preprint_remapped.ckpt
├── rf3_foundry_09_21_preprint_remapped.ckpt
├── rfd3_latest.ckpt
├── rfd3na-1190.ckpt
├── rfd3na-890.ckpt
├── proteinmpnn_v_48_020.pt
├── ligandmpnn_v_32_010_25.pt
└── solublempnn_v_48_020.pt
```

Foundry CLI 会搜索 `~/.foundry/checkpoints` 和 `$FOUNDRY_CHECKPOINT_DIRS`。也可以在命令中显式传入 `ckpt_path`。

## 六、推理示例

### 6.1 RF3 结构预测

```bash
cd /workspace/foundry

rf3 fold \
  inputs="$(pwd)/models/rf3/tests/data/5vht_from_json.json" \
  out_dir=/workspace/foundry_outputs/rf3_5vht \
  ckpt_path=/weights/foundry/rf3_foundry_01_24_latest_remapped.ckpt \
  skip_existing=False
```

若输入 JSON 中包含相对 MSA 路径，请从 Foundry 仓库根目录运行，或把 JSON 中路径改为绝对路径。

### 6.2 RFD3 设计

```bash
cd /workspace/foundry

rfd3 design \
  out_dir=/workspace/foundry_outputs/rfd3_demo \
  inputs=models/rfd3/docs/examples/demo.json \
  ckpt_path=/weights/foundry/rfd3_latest.ckpt \
  skip_existing=False \
  dump_trajectories=False \
  prevalidate_inputs=True
```

### 6.3 RFD3NA 设计

```bash
cd /workspace/foundry

cat > /workspace/rfd3na_paired_position.json <<EOF
{
  "paired_position_input_ss": {
    "paired_position_list": [
      "A3,B3",
      "A5,B5",
      "A7,B7",
      "A9,B9",
      "A11,B11",
      "A13,B13",
      "A15,B15",
      "A17,B17",
      "A19,B19"
    ],
    "contig": "20-20R,/0,20-20R",
    "length": "40-40",
    "input": "$(pwd)/models/rfd3na/docs/input_pdbs/AMP.pdb"
  }
}
EOF

rfd3na design \
  out_dir=/workspace/foundry_outputs/rfd3na_tiny \
  inputs=/workspace/rfd3na_paired_position.json \
  ckpt_path=/weights/foundry/rfd3na-1190.ckpt \
  skip_existing=False \
  dump_trajectories=False \
  prevalidate_inputs=True \
  read_sequence_from_sequence_head=False
```

### 6.4 ProteinMPNN / LigandMPNN

Foundry 中的 MPNN CLI 与权重 API 仍在快速变化。使用原始 ProteinMPNN/LigandMPNN 权重时，需要设置 `--is_legacy_weights True`。

ProteinMPNN 最小固定骨架设计：

```bash
cd /workspace/foundry

mpnn \
  --model_type protein_mpnn \
  --checkpoint_path /weights/foundry/proteinmpnn_v_48_020.pt \
  --is_legacy_weights True \
  --structure_path models/rf3/tests/data/5vht_from_file.cif \
  --out_directory /workspace/foundry_outputs/mpnn_protein \
  --name 5vht_mpnn \
  --batch_size 1 \
  --number_of_batches 1 \
  --write_fasta True \
  --write_structures True
```

LigandMPNN 最小配体上下文设计：

```bash
cd /workspace/foundry

mpnn \
  --model_type ligand_mpnn \
  --checkpoint_path /weights/foundry/ligandmpnn_v_32_010_25.pt \
  --is_legacy_weights True \
  --structure_path models/rf3/tests/data/glke_with_ligands_from_cif.cif \
  --out_directory /workspace/foundry_outputs/mpnn_ligand \
  --name glke_ligand_mpnn \
  --batch_size 1 \
  --number_of_batches 1 \
  --write_fasta True \
  --write_structures True
```

上述两个 MPNN 示例已验证会分别生成 `.fa` 和 `.cif`。如果需要稳定基准测试，建议按上游 README 的提示使用原始 ProteinMPNN/LigandMPNN 仓库；Foundry 重新实现版本更适合与 RFD3/RF3 流水线组合使用。

## 七、cuequivariance 与 atomworks[ml] 限制

当前 MACA PyTorch 镜像不支持 NVIDIA `cuequvariance`/`cuequivariance`。Foundry 的绕过方式是：

- 不安装 `rc-foundry[rf3]` 或 `rc-foundry[all]`。
- 安装源码前删除 `cuequivariance_ops_cu12`、`cuequivariance_ops_torch_cu12`、`cuequivariance_torch` 相关行。
- 设置 `DISABLE_CUEQUIVARIANCE=1`。
- 对 `atomworks[ml]` 额外依赖组保持谨慎；如果它引入仅面向 GPU 的依赖、触发 `torch` 依赖解析或安装 `cuequivariance`，应改为安装 `atomworks` 基础包或手动拆分依赖。

Foundry 的 RFD3/RFD3NA 注意力代码在 cueq 导入失败时会回退到 PyTorch 路径；RF3 也会读取 `DISABLE_CUEQUIVARIANCE` 来关闭 cueq。

RFD3/RFD3NA 还会使用 `apex.normalization.fused_layer_norm.FusedRMSNorm`。当前 `MultiDimLinear` 的 `out_features = np.prod(out_shape)` 会把 NumPy 标量传入 RMSNorm shape，导致 MACA 基础镜像中的 apex 扩展在 bfloat16 推理时触发 `rms_forward_affine()` 参数类型不兼容错误；本文档的 Dockerfile 将其改为 `out_features = int(np.prod(out_shape))`，保留上游 apex FusedRMSNorm 路径。

## 八、Dockerfile 使用

准备源码：

```bash
cd /path/to/project-root
mkdir -p third_party
git clone https://github.com/RosettaCommons/foundry.git third_party/foundry
git -C third_party/foundry checkout 4010e3e2e735
```

构建并运行：

```bash
docker build \
  -f LifeScience/Foundry/Dockerfile \
  --build-arg FOUNDRY_SOURCE=third_party/foundry \
  -t foundry:maca-torch2.10 \
  .

docker run -it --rm \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=64G \
  -v /path/to/foundry-weights:/weights/foundry:ro \
  -v /path/to/workspace:/workspace \
  foundry:maca-torch2.10
```

## 九、维护与支持

沐曦仅维护本文档中的 MACA 环境配置说明。Foundry 的模型架构、Hydra 配置、输入 schema、权重注册表和 notebook 流水线请参考：

- [Foundry GitHub](https://github.com/RosettaCommons/foundry)
- [RF3 README](https://github.com/RosettaCommons/foundry/tree/main/models/rf3)
- [RFD3 README](https://github.com/RosettaCommons/foundry/tree/main/models/rfd3)
- [RFD3NA README](https://github.com/RosettaCommons/foundry/tree/main/models/rfd3na)
- [MPNN README](https://github.com/RosettaCommons/foundry/tree/main/models/mpnn)

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码；必要的本地安装修改已在文中列出。您按照本文档配置、部署或使用相关软件时，应遵守使用许可证的条款及条件。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
