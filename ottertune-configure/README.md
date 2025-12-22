# OtterTune 复现与环境配置指导手册

本手册记录了在 Linux 环境下复现 OtterTune 核心算法（GPRGD）的完整过程，包括环境安装、依赖解决、以及运行中遇到的 Bug 和解决方法。

## 1. 环境准备

OtterTune 的版本较为陈旧，主要依赖 Python 3.6 和 TensorFlow 1.12.2。为了避免与现有的 PyTorch 环境冲突，**必须**创建一个独立的 Conda 环境。

### 创建 Conda 环境
```bash
conda create -n ottertune python=3.6
conda activate ottertune
```

### 安装依赖项
OtterTune 的依赖项较多，安装过程中可能会遇到网络极慢或中断的问题。

**关键经验：**
如果发现下载进度卡住（尤其是下载 TensorFlow 大包时），通常是因为系统代理干扰了国内镜像源的访问。

**解决方法：**
在执行 `pip` 前，务必先取消当前会话的代理设置：

```bash
# 进入仓库目录
cd /home/dyx/VDTuner/ottertune-configure/ottertune

# 核心：在一行命令中禁用代理并使用清华源安装
unset http_proxy https_proxy && pip install -r server/website/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装对接 VDTuner 所需的额外依赖
pip install pyyaml joblib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这么做？**
- `unset http_proxy https_proxy`: 确保连接不走代理，直达国内服务器。
- `-i ...`: 使用清华镜像加速。
- `&&`: 确保这两个操作在同一个执行环境下生效。

**关键依赖版本确认：**
- `tensorflow==1.12.2`
- `numpy<1.17.0` (TF 1.12 的要求)
- `scikit-learn`

---

### 安装benchmark相关的pip包

由于我们切换了一个全新的conda环境，之前安装过的benchmark相关的pip包都不见了，所以需要重新安装。否则，跑benchmark的run.py的时候会运行失败。
我在`/home/dyx/VDTuner/ottertune-configure/requirements.txt`当中列出了所有的benchmark相关的pip包，可以根据这个文件重新安装。
提示：这个文件中的pip包和之前环境的pip包可能有所不同，这是因为我们安装的`tensorflow`相关的python环境太老了，相关的依赖也会有所变化。

```txt
stopit
typer
tqdm
elasticsearch
pymilvus
opensearch-py
redis
weaviate
```

### 移除benchmark中的冗余依赖

#### 问题背景
在运行benchmark时，我们发现了两个主要问题：
1. **Qdrant依赖缺失**：代码中引用了Qdrant客户端，但我们的环境中没有安装
2. **Weaviate兼容性问题**：Weaviate客户端依赖`importlib.metadata`，而Python 3.6版本不支持该模块

由于我们的实验只需要使用Milvus向量数据库，因此可以安全地移除这些冗余的数据库依赖。

#### 解决方案
修改`/home/dyx/VDTuner/vector-db-benchmark-master/engine/clients/client_factory.py`文件，注释掉以下内容：

```python
# 注释掉Qdrant相关导入
# from engine.clients.qdrant import QdrantConfigurator, QdrantSearcher, QdrantUploader

# 注释掉Weaviate相关导入
# from engine.clients.weaviate import (
#     WeaviateConfigurator,
#     WeaviateSearcher,
#     WeaviateUploader,
# )

# 在配置字典中注释掉Qdrant和Weaviate
ENGINE_CONFIGURATORS = {
    # "qdrant": QdrantConfigurator,
    # "weaviate": WeaviateConfigurator,
    "milvus": MilvusConfigurator,
    # ... 其他保留的数据库
}

ENGINE_UPLOADERS = {
    # "qdrant": QdrantUploader,
    # "weaviate": WeaviateUploader,
    "milvus": MilvusUploader,
    # ... 其他保留的数据库
}

ENGINE_SEARCHERS = {
    # "qdrant": QdrantSearcher,
    # "weaviate": WeaviateSearcher,
    "milvus": MilvusSearcher,
    # ... 其他保留的数据库
}
```

#### 验证是否可以跑benchmark了
完成修改后，可以去到`vector-db-benchmark-master`，通过以下命令验证Milvus是否能正常运行：
```bash
python run.py --engines milvus-p10 --datasets random-100 --host 127.0.0.1
```

提示：运行上面指令的前提是启动了milvus服务，并且没有关闭它。
手动启动的方法：进入到`~/VDTuner/vector-db-benchmark-master/engine/servers/milvus-single-node$`执行过`docker compose up -d`，手动把milvus服务启动，否则会报错（跑数据集依赖于milvus服务）。

如果看到以下输出，说明修改成功：
```
(ottertune) dyx@server9050:~/VDTuner/vector-db-benchmark-master$ python run.py --engines milvus-p10 --datasets random-100 --host 127.0.0.1
Running experiment: milvus-p10 - random-100
established connection
/home/dyx/VDTuner/vector-db-benchmark-master/datasets/random-100 already exists
Experiment stage: Configure
Experiment stage: Upload
100it [00:00, 11644.70it/s]
Upload time: 1.677040726877749
Total import time: 9.320258591789752
Experiment stage: Search
10it [00:00, 3788.55it/s]
Experiment stage: Done
Results saved to:  /home/dyx/VDTuner/vector-db-benchmark-master/results
(ottertune) dyx@server9050:~/VDTuner/vector-db-benchmark-master$ 
```
注：上述的环境是ottertune，这表明，我们在一个兼容ottertune的环境中，跑通了benchmark。



