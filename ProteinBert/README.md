# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

# 沐曦GPU运行ProteinBERT说明文档

## 一、ProteinBERT简介

[ProteinBERT](https://github.com/nadavbra/protein_bert) 是一个蛋白质语言模型，在 UniRef90 中约 1.06 亿个蛋白质序列上进行了预训练。该预训练模型可以在几分钟内针对任何与蛋白质相关的任务进行微调。ProteinBERT 在多种基准测试中均达到了领先水平，ProteinBERT基于 Keras/TensorFlow 框架构建。

ProteinBERT 的深度学习架构受 BERT 启发，但包含多项创新，例如具有线性序列长度复杂度的全局注意力层（相比于自注意力机制的平方 / n² 增长）。因此，该模型能够处理几乎任意长度的蛋白质序列，包括极长的蛋白质序列（长度超过数万个氨基酸）。

## 二、沐曦GPU环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的[MACA TensorFlow2镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=tensorflow2:2.13.1-maca.ai3.7.0.6-py310-ubuntu22.04-amd64)启动容器：

```bash
docker run -it --name test-proteinbert \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/user:/workspace \
  cr.metax-tech.com/public-ai-release/maca/tensorflow2:2.13.1-maca.ai3.7.0.6-py310-ubuntu22.04-amd64 \
  /bin/bash
```

**参数说明：**
- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦GPU设备
- `--group-add video`：添加video组以访问GPU
- `--shm-size=4G`：设置共享内存大小
- `-v`：挂载工作目录

> 更多更新的基础镜像，请移步[沐曦开发者|镜像资源中心](https://developer.metax-tech.com/softnova/docker/)

### 2.2 验证环境

进入容器后，验证TensorFlow环境：

```bash
root@afb6339a7ae3:/workspace# pip list | grep tensor
```

输出显示已安装沐曦定制版TensorFlow：
```
tensorflow                   2.13.1+maca3.7.0.6
```

### 2.3 安装ProteinBERT

#### 2.3.1 安装依赖包

```bash
# 升级pip
conda run -n base pip install --upgrade pip

# 下载并安装ProteinBERT
cd /opt && git clone --recursive https://github.com/nadavbra/protein_bert.git
conda run -n base pip install /opt/protein_bert
```

安装成功输出示例：
```
Successfully installed protein-bert-1.0.1
```

### 2.4 预训练模型准备

ProteinBERT首次运行时会自动下载预训练权重文件，默认保存到 `/root/proteinbert_models/default.pkl`。

**方式一：自动下载（首次运行）**

模型会自动从网络下载，但可能较慢或需要稳定网络连接。

**方式二：手动配置预训练模型**

若已预先下载模型文件（如 `epoch_92400_sample_23500000.pkl`），可通过软链接方式配置：

```bash
# 删除可能存在的默认文件（如有）
rm -f /root/proteinbert_models/default.pkl

# 创建模型目录
mkdir -p /root/proteinbert_models/default

# 创建软链接指向预训练模型文件
ln -s /workspace/epoch_92400_sample_23500000.pkl /root/proteinbert_models/default.pkl

```

### 2.5 编写测试脚本

创建 `test_proteinbert.py`：

```python
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import numpy as np
# 示例蛋白质序列
seqs = ["SMTDLLSAEDIKKAIGAFTAADSFDHKKFFQMVGLKKKSADDVKKVFHILDKDKDGFIDEDELGSILKGFSSDARDLSAKETKTLMAAGDKDGDGKIGVEEFSTLVAES","MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
seq_len = 120

#载入预训练模型并进行预测（模型已经提前下载至/root/proteinbert_models/）
pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))

# 对序列进行编码
encoded_x = input_encoder.encode_X(seqs,seq_len)
batch_size = 1
# 模型预测
local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size) #warmup
local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
print({global_representations.shape},{local_representations.shape})
```

### 2.6 运行测试

```bash
python test_proteinbert.py
```

**运行输出说明：**

1. **CUDA环境初始化日志**（正常现象）：
```
2026-06-09 18:36:26.598773: W tensorflow/core/common_runtime/gpu/gpu_device.cc:2052] 
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 8.0. 
CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
```
这表明TensorFlow正在为沐曦GPU进行JIT编译，首次运行时可能需要较长时间（30分钟以上），后续运行会使用缓存加速。

2. **cuDNN前端禁用日志**（正常现象）：
```
Disabling cuDNN frontend for the following convolution: 
  input: {count: 1 feature_map_count: 128 spatial: 1 120 ...}
  ... because the current cuDNN version does not support it.
```
这说明某些卷积操作未使用cuDNN前端，但会使用TensorFlow内部算子替代，不影响功能。

3. **模型推理输出**：
```
2/2 [==============================] - 8s 105ms/step
2/2 [==============================] - 0s 7ms/step
{(2, 15599)} {(2, 120, 1562)}
```

其中：
- **global_representations**：形状 `(batch_size, 15599)`，表示每个序列的全局特征向量
- **local_representations**：形状 `(batch_size, seq_len, 1562)`，表示每个序列中每个位置的局部特征向量


> 更多示例，请开发者自行查阅[ProteinBERT官方](https://github.com/nadavbra/protein_bert)。
## 三、Dockerfile示例

```Dockerfile
FROM cr.metax-tech.com/public-ai-release/maca/tensorflow2:2.13.1-maca.ai3.7.0.6-py310-ubuntu22.04-amd64

WORKDIR /workspace

# 安装必要工具
RUN apt-get update && apt-get install -y git wget && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 设置conda环境
ENV PATH="/opt/conda/bin:${PATH}"

# 安装ProteinBERT
RUN conda run -n base pip install --upgrade pip
RUN cd /opt && git clone --recursive https://github.com/nadavbra/protein_bert.git
RUN cd /opt/protein_bert && conda run -n base pip install .

# 可选：预下载或复制模型文件到默认位置
# wget -q -O /root/proteinbert_models/default.pkl https://github.com/nadavbra/proteinbert_data_files/blob/master/epoch_92400_sample_23500000.pkl 

# 设置入口命令
CMD ["/bin/bash"]
```

## 四、维护与支持

沐曦仅维护本说明文档中的环境配置部分，更多ProteinBERT的详细用法请参考：
- [ProteinBERT官方GitHub](https://github.com/nadavbra/protein_bert)
- [ProteinBERT论文](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274)

如遇沐曦GPU环境相关问题，可在[沐曦开发者论坛](https://developer.metax-tech.com/forum/)留言咨询。