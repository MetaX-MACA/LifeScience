# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

# 沐曦GPU运行DeepFRI说明文档

## 一、DeepFRI简介

[DeepFRI](https://github.com/flatironinstitute/DeepFRI)（Deep Functional Residue Identification）是一个基于深度学习的蛋白质功能预测工具，通过结合图卷积神经网络（Graph Convolution Networks，GCN）和蛋白质语言模型（Protein Language Model）与蛋白质结构信息来预测蛋白质功能。该方法在生物信息学领域具有重要意义，尤其是在基因本体（Gene Ontology, GO）和酶分类（Enzyme Commission, EC）任务中表现出色。


## 二、沐曦GPU环境配置与运行

### 2.1 环境准备

使用沐曦开发者社区提供的[MACA TensorFlow2镜像](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=tensorflow2:2.13.1-maca.ai3.7.0.6-py310-ubuntu22.04-amd64)启动容器：

```bash
docker run -it --name test-deepfri \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /remote_home/zhli/ai4s-opensource:/workspace \
  cr.metax-tech.com/public-ai-release/maca/tensorflow2:2.13.1-maca.ai3.7.0.6-py310-ubuntu22.04-amd64 \
  /bin/bash
```

**参数说明：**
- `--device=/dev/mxcd --device=/dev/dri`：挂载沐曦GPU设备
- `--group-add video`：添加video组以访问GPU
- `--shm-size=4G`：设置共享内存大小
- `-v`：挂载工作目录

> 可根据需要使用更新版本镜像，请移步[沐曦开发者|镜像资源中心](https://developer.metax-tech.com/softnova/docker/)


### 2.2 验证环境

进入容器后，验证TensorFlow环境：

```bash
root@db62ca9dfd78:/workspace# pip list | grep tensor
```

输出显示已安装沐曦定制版TensorFlow：
```
tensorflow                   2.13.1+maca3.7.0.6
```

### 2.3 安装依赖包

DeepFRI的`setup.py`文件中列出了依赖包，但各依赖的版本偏低，这里手动安装础镜像中未包含的`scikit-learn`和`biopython`：

```bash
# 升级pip（可选）
conda run -n base pip install --upgrade pip

# 安装所需依赖
conda run -n base pip install scikit-learn biopython matplotlib
```

安装成功输出：
```
Successfully installed biopython-1.87 joblib-1.5.3 scikit-learn-1.7.2 threadpoolctl-3.6.0
```

### 2.4 安装DeepFRI

#### 2.4.1 下载DeepFRI
```bash
git clone https://github.com/flatironinstitute/DeepFRI.git:
cd DeepFRI
git checkout 3bd6c7eae67894ccedbe1117e33b3abe544bca71
```

#### 2.4.2 准备预训练数据
```bash
cd DeepFRI
wget https://users.flatironinstitute.org/~renfrew/DeepFRI_data/trained_models.tar.gz
tar -zxvf trained_models.tar.gz
```

#### 2.4.3 安装
进入DeepFRI目录修改setup.py，注释install_requires的实际内容，安装DeepFRI。
```python
      install_requires=[
                        # 'numpy==1.18.5',
                        # 'tensorflow-gpu==2.3.1',
                        # 'networkx==2.4',
                        # 'scikit-learn==0.23.1',
                        # 'biopython==1.76',
                        ],

conda run -n base pip install .
```

安装成功输出：
```
Successfully built DeepFRI
Successfully installed DeepFRI-0.0.1
```

## 三、推理预测

推理主要调用`predict.py`脚本，请参考[protein-function-prediction](https://github.com/flatironinstitute/DeepFRI#protein-function-prediction)

部分曦云C500结果如下：

```bash
# option-2-predicting-functions-of-a-protein-from-its-sequence
python predict.py --seq 'SMTDLLSAEDIKKAIGAFTAADSFDHKKFFQMVGLKKKSADDVKKVFHILDKDKDGFIDEDELGSILKGFSSDARDLSAKETKTLMAAGDKDGDGKIGVEEFSTLVAES' -ont mf --verbose
```

输出：

| 蛋白质标识 | GO-term/EC-number | 得分 | GO-term/EC-number名称 |
|-----------|---------|------|--------------|
| query_prot | GO:0005509 | 0.99768 | calcium ion binding |

---

```bash
# option-3-predicting-functions-of-proteins-from-a-fasta-file
python predict.py --fasta_fn examples/pdb_chains.fasta -ont mf -v
```

输出（示例，已列出多个蛋白的预测结果，包括谷胱甘肽转移酶活性、DNA 结合、GTP 结合等）：

| 蛋白质标识 | GO-term/EC-number | 得分 | GO-term/EC-number名称 |
|--------|---------|------|------|
|1S3P-A| GO:0005509| 0.99768| calcium ion binding
|2J9H-A| GO:0004364| 0.46954| glutathione transferase activity
|2J9H-A| GO:0016765| 0.19914| transferase activity, transferring alkyl or aryl (other than methyl) groups
|2J9H-A| GO:0097367| 0.10541| carbohydrate derivative binding
|2PE5-B| GO:0003677| 0.53504| DNA binding
|2W83-E| GO:0032550| 0.99261| purine ribonucleoside binding
| ... | ... | ... | ... |

---

> 由于部分tensorflow for maca部分功能还未完整适配和测试，部分sample会出现运行问题，tensorflow框架正在持续推进中。


## 四、常见问题与注意事项

### 4.1 依赖版本兼容性
原DeepFRI的`setup.py`要求`tensorflow-gpu==2.3.1`，但在沐曦GPU环境中使用`tensorflow==2.13.1`可以正常运行，无需修改。

### 4.2 预训练模型
首次运行时，DeepFRI不会自动下载预训练权重文件。需要提前下载预训练文件，并解压到DeepFRI目录。

### 4.3 批量预测
除了单序列预测，DeepFRI还支持：
- 从FASTA文件批量预测：`python predict.py --fasta input.fasta -ont mf`
- 从PDB结构文件预测：`python predict.py --pdb 1ake.pdb -ont mf`

### 4.4 模型编译日志
```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 8.0. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
Disabling cuDNN frontend for the following convolution: ...
```
这些日志表明TensorFlow正在为各种卷积核尺寸进行JIT编译（即时编译），卷积格式不匹配导致`frontend`被禁用，但是会使用tensorflow内部的算子，这是正常现象。


## 五、Dockerfile示例

```Dockerfile
FROM cr.metax-tech.com/public-ai-release/maca/tensorflow2:2.13.1-maca.ai3.7.0.6-py310-ubuntu22.04-amd64
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda run -n base pip install --upgrade pip
RUN conda run -n base pip install networkx scikit-learn biopython
RUN cd /opt && git clone https://github.com/flatironinstitute/DeepFRI.git
RUN cd /opt/DeepFRI && sed -i '/install_requires=.*\[/,/],/c\      install_requires=[],' setup.py
RUN cd /opt/DeepFRI && conda run -n base pip install .
```

## 六、其他
1. 开发者在初次使用曦云GPU运行DeepFRI时，可遵循本手册进行测试。
2. 在曦云GPU的运行DeepFRI，依赖沐曦提供的tensorflow镜像，开发者亦可以在开发者社区提供的基础镜像上直接安装沐曦提供的tensorflow whl包进行测试。
3. 沐曦仅维护部分case的正确性，如出现运行问题，可提交issue，亦可以在开发者社区提交bug反馈。
4. 了解更多沐曦开源项目，请参考[沐曦开源社区](https://github.com/metax-maca)
