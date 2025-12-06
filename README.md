## VDTuner复现指导手册

[TOC]

### 假设和心理预期

**假设**：已经使用**vscode/AI IDE**连接上了**linux**服务器，linux服务器**常见工具**和**conda**配置好了，也能正常**联网**，这一步不再演示。同时，我们假设所有的python库都安装好了，从0复现的时候，需要手动去安装一些库，例如**botorch**，这里不再演示。
**预期**：按照手册配置环境大约需要**1**个小时，如果运气好(bug少)可能更短一些，运行完整实验(GloVe 数据集，200次迭代)大约需要30000秒，即**8.33小时**，即半天，不过，我们先只需要能跑起来即可，无需跑通。

**关键难点：**服务器没有sudo权限（需要使用无sudo的命令），而且时不时连不上GitHub等网站（有时候需要手动下载）。

### 下载github仓库(预计，1分钟都不到)

选择一个合适的路径，建议放在/home/目录，这样的好处是大部分路径都和GitHub仓库的一致，只需要把github仓库中的`/home/ytn/仓库名`换成`/home/xxx/VDTuner`即可.

```shell
git clone https://github.com/tiannuo-yang/VDTuner
```

![image-20251125100457535](./assets/image-20251125100457535.png)

### 项目结构介绍

VDTuner包含两个文件夹，一个是auto-config文件夹，也就是我们的vdtuner，负责调优。另一个是benchmark文件夹，负责跑benchmark。

这个benchmark来自**另一个**仓库，而不是VDTuner自己制作的，它的链接：`https://qdrant.tech/benchmarks/`。它不仅可以跑不同的数据集，而且可以**选不同的向量数据库**,也就是**引擎（Engine）**，在VDTuner当中选择的是**milvus**而不是其他向量数据库，选择单机场景（**single-node**）而非分布式。

简单来说**VDTuner需要调用benchmark**来评估MOBO模型的好坏。

```text
.
├── auto-configure/                 <-- [大脑] VDTuner 的核心代码目录
│   ├── configure.py                <-- [关键] 配置文件，这里必须填所有的绝对路径！
│   ├── vdtuner/
│   │   ├── main_tuner.py           <-- [主程序] 整个优化的入口，修改迭代次数就在这
│   │   └── utils.py                <-- [工具] 负责调用 benchmark 脚本，也含有绝对路径配置
│   ├── index_param.json            <-- 索引参数搜索空间
│   ├── system_param.json           <-- 系统参数搜索空间
│   └── whole_param.json            <-- 汇总的搜索空间
│
└── vector-db-benchmark-master/     <-- [手脚] 负责实际跑 Milvus 和发请求
    ├── run_engine.sh               <-- [执行脚本] VDTuner 通过命令行调用这个脚本来跑测试
    ├── run.py                      <-- Python入口，解析命令行参数
    ├── datasets/                   <-- [数据] 数据集下载后存放的位置
    │   └── random-100/             <-- 自带的微型测试数据
    ├── engine/
    │   └── servers/
    │       └── milvus-single-node/ <-- [服务端] Milvus 的 Docker 配置
    │           ├── docker-compose.yml
    │           ├── milvus.yaml     <-- [目标] VDTuner 会不断修改这个文件来调优
    │           └── milvus.yaml.backup <-- [备份] 原始配置，用于恢复
    └── experiments/
        └── configurations/
            └── milvus-single-node.json <-- [实验配置] 定义并发数、数据集参数
```

### docker下载镜像

首先进入到milvus-single-node，意思是，我们要跑单机的milvus。

```shell
cd ~/VDTuner/vector-db-benchmark-master/engine/servers/milvus-single-node
```

![image-20251125113258315](./assets/image-20251125113258315.png)

其中应该有3个文件，分别是：`docker-compose.yml  milvus.yaml  milvus.yaml.backup`。

其中VDTuner 会**不断修改**milvus.yaml文件来调优，打开milvus.yaml就可以看到索引、参数，计算相似度的方法。

如果 VDTuner 决定把索引换成 **IVF_FLAT**，它就会把这一行改成 `{"nlist": 1024, ...}`。

![image-20251125120353550](./assets/image-20251125120353550.png)

为了能着恢复默认的.yaml文件，这里有一个备份`milvus.yaml.backup`，如果没有备份过，需要手动备份一下`cp milvus.yaml milvus.yaml.backup`。

关键是来看`docker-compose.yml`，首先我们需要改其中`volumes`的路径，把`/home/ytn/仓库名`换成`/home/xxx/VDTuner`，不然路径出错了。

![image-20251125121417462](./assets/image-20251125121417462.png)

```yml
/home/dyx/VDTuner
```

后面的指令执行后，会有一个`volumes`文件夹出现在`milvus-single-node`文件夹当中。

![image-20251204145211783](./assets/image-20251204145211783.png)

接下来，需要执行下面的指令来**启动 Milvus 服务**。确保执行的路径中含有`docker-compose.yml`文件，其中milvus的配置由`milvus.yaml`指定。

```shell
docker compose up -d
```

*   **up**: 启动。它会自动完成三个动作：下载镜像（Pull）、创建容器（Create）、启动容器（Start）。
*   **-d**: **Detached (后台运行)**。
    *   如果你不加 `-d`，容器的日志会直接霸占你的屏幕，你一按 `Ctrl+C`，容器就挂了。
    *   加了 `-d`，它就在后台默默工作，把控制权还给你。

然而，这条命令可能会遇到一些错误：

**情况1：之前的容器还有残留**，需要手动停止，之后执行

```shell
docker compose down -v
```

1.  **执行 `docker compose down -v`：** 
    *   **后果：** 存储在 Milvus 容器外部的**所有数据卷**都会被删除。这意味着存储在这些数据卷中的**所有向量数据(如，词向量妈妈[20,30,8...])、元数据和索引(如表名、向量维度)都会被永久清除**。
    *   **结果：** 重新 `docker compose up` 启动后，Milvus 数据库将是一个**全新的、空的状态**。

2.  **执行 `docker compose down`（不加 `-v`）：**
    *   **后果：** 容器和网络会被停止和删除，但是关联的**数据卷会被保留下来**。
    *   **结果：** 重新 `docker compose up` 启动后，Milvus 会重新挂载（re-mount）之前的数据卷，因此**原有的向量数据和元数据都会保留**，服务状态得以恢复。

**情况2：连不上网**

可以检查是否能连接google

```shell
curl www.google.com
```

如果不行，需要使用本地代理(如何获取代理不说了)，指定代理端口为自己的代理端口，例如7890，我这里是33210

本地端：

```shell
ssh -vvv -N -R 33210:localhost:33210 -p 端口号 用户名@ip地址
```

服务器端：

```shell
export http_proxy=http://127.0.0.1:33210; #HTTP
export https_proxy=http://127.0.0.1:33210; #HTTPS
```

之后检查，发现可以联网：

![image-20251125124854222](./assets/image-20251125124854222.png)

**情况3：可以联网，但是镜像就是下不下来**

原因分析：虽然终端(shell)里的代理已经修改了，然而，后台的**守护进程**（真正干活的）代理没有修改，除非去修改doker配置文件，然而这很繁琐

说人话：就算是`curl www.google.com`成功了，**如果本来不行现在还是不行。**

解决方案，使用类似于**毫秒镜像**(收费不贵，先去毫秒镜像`https://1ms.run/`把这个搞定)这样的网站作为下载源。

![image-20251125133017341](./assets/image-20251125133017341.png)

登录毫秒镜像，生成一个密钥。，然后服务器shell中登录。

```shell
docker login docker.1ms.run -u 1ms -p [你的毫秒镜像密钥]
```

然后把`docker-compose.yml`中的三个img换成毫秒镜像的通道。

![image-20251125133329943](./assets/image-20251125133329943.png)

![image-20251125133344157](./assets/image-20251125133344157.png)

![image-20251125133407868](./assets/image-20251125133407868.png)

```yml
image: quay.1ms.run/coreos/etcd:v3.5.5
image: docker.1ms.run/minio/minio:RELEASE.2023-03-20T20-16-18Z
image: docker.1ms.run/milvusdb/milvus:v2.3.1
```

再次执行`docker compose up -d`，会发现成功了。

![image-20251125132359057](./assets/image-20251125132359057.png)

之后执行

```shell
 docker compose ps
```

![image-20251125133921514](./assets/image-20251125133921514.png)

发现状态栏都是UP，说明容器健康

### 跑通最小的benchmark：random-100

我们需要跑通`run_engine.sh`，而`run_engine.sh`的核心就是运行`run.py`，**简单说，我们需要跑一个脚本，这个脚本中使用python运行run.py。**

然而，这里的`run.py`当中其实有一个雷，就是这句话，这里想要指定作者的python路径，就会找不到，注释之后，就可以自动寻找我们的路径。

```python
# sys.path.append('/home/ytn/.local/lib/python3.11/site-packages')
```

![image-20251125185111085](./assets/image-20251125185111085.png)

`run.py`是整个 benchmark 的入口。它**不关心具体的数据库细节**，只负责流程控制：

1. 读取配置（我们要测谁？**测什么数据？**）。
2. 下载数据。
3. 指挥具体的客户端去干活。

在正式开始调优之前，需要先测试一下能否跑数据集，然而，完整跑一整个数据集太慢了，所以我们指定

```shell
random-100
```

作为我们的数据集，相当于测试随机的100维向量。

首先去到数据集目录

```shell
cd ~/VDTuner/vector-db-benchmark-master
```

![image-20251125132633440](./assets/image-20251125132633440.png)

准备执行./run_engine.sh，需要加一个可执行权限（如果没有加）。

```shell
chmod +x ./run_engine.sh
```

![image-20251125134523350](./assets/image-20251125134523350.png)

>因为是在实验室服务器上，我**没有 sudo 权限**，所以原仓库里那种暴力重启 Docker、sudo 删除文件的脚本是跑不通的。
>
>同时，由于这份代码使用的是老版本docker，使用的是`docker-compose`，而我们是新版的docker，应该用`docker compose`，中间不用`-`而要用空格。
>
>这里写了一个更优化的版本，去掉了sudo。首先关闭milvus，删掉之前插入的向量，然后根据milvus.yaml，重启milvus，然后开始测试数据集。

将下面的脚本`run_engine_test.sh`放到`run_engine.sh`相同目录

赋予执行权限，然后执行

```shell
chmod +x ./run_engine_test.sh
./run_engine_test.sh
```

需要拷贝的脚本

```shell
#!/usr/bin/env bash
set -e

# 获取脚本所在目录
SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
# 默认值设置
SERVER_PATH=${1:-"milvus-single-node"}
ENGINE_NAME=${2:-"milvus-p10"}
DATASETS=${3:-"random-100"} # 默认先用 random-100 跑通
SERVER_HOST="127.0.0.1"

# 定义 Milvus 目录
MILVUS_DIR="$SOURCE_DIR/engine/servers/$SERVER_PATH"
MONITOR_DIR="$SOURCE_DIR/monitoring"

echo "======================================="
echo "🛠️  开始测试流程"
echo "Engine: $ENGINE_NAME | Dataset: $DATASETS"
echo "======================================="

# 1. 启动 Docker 资源监控 (后台运行)
# 注意：确保 monitor_docker.sh 有执行权限
echo ">>> [Step 1] 启动后台监控..."
if [ -f "$MONITOR_DIR/monitor_docker.sh" ]; then
    # 清理旧日志
    rm -f "$MONITOR_DIR/docker.stats.jsonl"
    # 后台运行
    nohup bash -c "cd $MONITOR_DIR && ./monitor_docker.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "    监控进程 PID: $MONITOR_PID"
else
    echo "⚠️  未找到监控脚本，跳过监控步骤。"
fi

# 2. 重置 Milvus 环境 (Down -> Clean -> Up)
echo ">>> [Step 2] 重置 Milvus..."
cd "$MILVUS_DIR"
docker compose down -v  # 停止并删卷
sleep 5                 # 稍微缓冲一下

# 启动容器
docker compose up -d

# 3. 等待启动 (你的经验数据：90s，这里为了测试可以用短一点，比如 random-100 可能 30s 就够)
echo ">>> [Step 3] 等待服务启动 (90s)..."
sleep 90

# 4. 运行 Python 测试
echo ">>> [Step 4] 运行 Benchmark..."
# 代理设置
export no_proxy="localhost,127.0.0.1,::1"

# 切换回根目录运行脚本
cd "$SOURCE_DIR"
# 这里的 python 路径按你服务器实际情况写
python run.py --engines "$ENGINE_NAME" --datasets "${DATASETS}" --host "$SERVER_HOST"

# 5. 测试结束，停止监控和容器
echo ">>> [Step 5] 收尾工作..."

# 杀掉监控进程
if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID || true
    # 移动监控日志
    mkdir -p "$MONITOR_DIR/results"
    # 构造文件名
    LOG_NAME=$(echo "$ENGINE_NAME" | sed -e 's/[^A-Za-z0-9._-]/_/g')
    mv "$MONITOR_DIR/docker.stats.jsonl" "$MONITOR_DIR/results/${LOG_NAME}-docker.stats.jsonl" 2>/dev/null || true
    echo "    监控日志已保存。"
fi

# 停止容器 (可选，如果你想保留现场查看日志，可以注释掉这行)
# cd "$MILVUS_DIR" && docker compose down

# 6. 打印结果
echo "======================================="
echo "📊 测试结果摘要:"
# 获取最新的结果文件
RES_FILE=$(ls -t results/ | grep -v 'upload' | head -n 1)
if [ -n "$RES_FILE" ]; then
    cat "results/$RES_FILE" | grep -E "mean_precisions|rps|p95_time" | sed 's/.*: \([0-9.]*\),/\1/'
else
    echo "0 0 0"
fi
echo "======================================="
```

测试发现可以跑通：

<img src="./assets/image-20251203144918775.png" alt="image-20251203144918775" style="zoom: 80%;" />

提示：经过我的测试，docker启动milvus服务之后，等待90秒是必须的，只设60秒会导致连接失败

![image-20251204181504563](./assets/image-20251204181504563.png)

### 理解跑数据集的逻辑

首先是数据集，去到**datasets文件夹**当中，其中有所有下载的数据集。

可以查看`datasets.json`，其中有所有可以下载的数据集（都没有下载，需要先下载，然后才能跑数据集）

![image-20251204154955578](./assets/image-20251204154955578.png)

唯一下载好的是random-100数据集。

![image-20251204150444419](./assets/image-20251204150444419.png)

首先，random-100是一个直接放到了GitHub仓库中的默认数据集，不需要下载，没有下不下来的风险。

这个数据集当中有3个文件，第一个是**数据库向量**文件，也就是vectors.jsonl，其中有100个维度为100的向量。

然后是**查询向量文件`vectors.jsonl`**，总共有10个维度为100的查询向量。

最后是**标准答案**，也就是10个向量的最近邻居（K=1，所以每个向量的查询结果只有一个向量）

![image-20251204152941843](./assets/image-20251204152941843.png)

然后我们来看查询的结果，去到**results**文件夹当中，可以发现其中存储了一些结果，例如：
`milvus-p10-random-100-search-0-2025-12-03-06-42-53.json`

格式是**向量数据库名-数据集名-查询/构建-时间**

![image-20251204173815102](./assets/image-20251204173815102.png)

### 即使跑过，也再跑一遍

原版的代码中有一个小问题：就是，如果指定向量数据库（milvus），指定数据集（random100），只要跑过一次，就不会继续运行，而是会说，**结果已经存在了，所以不运行了。**
类似于这样的提示：

```shell
/home/dyx/VDTuner/vector-db-benchmark-master/datasets/random-100 already exists
Skipping run for milvus-p10 since it already ran 1 search configs previously
>>> [Step 5] 收尾工作...
```

然而，这是**不好的**，因为VDTuner进行向量数据库系统调优的时候，针对的是同一个数据库（milvus），而且是固定一个数据集，例如glove，不断修改milvus的配置，如果这样跑了一次就跑不了。

原版的`run_engine.sh`采取的方案是每次清空`results/*`文件夹，这样的问题是，每次的结果都被删除了。

因而，在我们的脚本`run_engine_test.sh`不选择去删除results/*文件夹，而是去修改一下python代码，无论results当中有什么东西，都重跑一次。

而修改这个东西也是特别简单的，只需要修改

```shell
/home/dyx/VDTuner/vector-db-benchmark-master/engine/base_client/client.py
```

当中的这一行，把skip_if_exists给成`False`

```python
        skip_if_exists: bool = False,
```

这样，即使结果存在，也不会跳过了。

![image-20251204180021666](./assets/image-20251204180021666.png)

### 明确论文中跑了哪几个数据集

首先要明确，VDTuner论文中总共跑了多少个数据集，总共3个核心数据集，作者还额外测评了2个，所以总共是5个。

其中第一个数据集的名字是：**glove-100-angular**

![image-20251204182559595](./assets/image-20251204182559595.png)

这是由于论文中提到了维度是100，而datasets.json当中有25、100的两个数据集，所以应该是100。

第二个数据集的名字是：**random-match-keyword-100-angular-no-filters**

![image-20251204183329110](./assets/image-20251204183329110.png)

这里有一个备选项，就是filters，在向量数据库当中，filters的意思是，不仅根据向量相似度查询，而且外加一些限制条件，进行过滤。

然而，VDTuner在测试纯粹的、无附加条件的向量搜索性能的数据集，所以应该是no-filters

第三个数据集的名字是：**random-geo-radius-2048-angular-no-filters**

同样，应该是no-fiters

![image-20251204183225975](./assets/image-20251204183225975.png)

第四个数据集的名称是：**arxiv-titles-384-angular-no-filters**

同样应该是**no-filters**

![image-20251204183622726](./assets/image-20251204183622726.png)

第五个数据集的名称是：**deep-image-96-angular**

![image-20251204183843797](./assets/image-20251204183843797.png)

### 跑通论文中的数据集glove-100-angular

我们需要运行脚本，这一次需要指定数据集

如果没有后缀，就是用milvus跑random100，但是强调数据集之后就是用milvus跑glove-100-angular

运行：

```shell
./run_engine_test.sh milvus-single-node milvus-p10 glove-100-angular
```

然而，有可能遇到这样的一个情况，就是超时了，卡在下载数据集了，解决方案是：使用本地代理，或者等个10分钟，可能需要比较长的时间

```shell
>>> [Step 3] 等待服务启动 (90s)...
>>> [Step 4] 运行 Benchmark...
Running experiment: milvus-p10 - glove-100-angular
established connection
Downloading http://ann-benchmarks.com/glove-100-angular.hdf5...
```

当然，如果就是下载不下来也不是没有可能，此时，需要手动下载，而手动下载的方法也不难

![image-20251204184756660](./assets/image-20251204184756660.png)

因为在`datasets.json`当中，每一个数据集都有一个下载链接，ctrl+左键单击，就可以用浏览器下载。

比如说点击这个链接http://ann-benchmarks.com/glove-100-angular.hdf5

![image-20251204184744271](./assets/image-20251204184744271.png)

下载到电脑之后，传到服务器的数据集文件夹下，确保以下文件夹下有hdf5文件

```shell
(torch) dyx@server9050:~/VDTuner/vector-db-benchmark-master/datasets/glove-100-angular$ ls
glove-100-angular.hdf5
```

![image-20251204185027882](./assets/image-20251204185027882.png)

然后重新跑就行了，就会看到类似于这样的结果：

```shell
```



![image-20251204185114677](./assets/image-20251204185114677.png)

### 跑通论文中的5个数据集

需要在`vector-db-benchmark-master`路径，也就是

```shell
(torch) dyx@server9050:~/VDTuner/vector-db-benchmark-master$ 
```

执行下面的指令

```shell
./run_engine_test.sh milvus-single-node milvus-p10 glove-100-angular
./run_engine_test.sh milvus-single-node milvus-p10 random-match-keyword-100-angular-no-filters
./run_engine_test.sh milvus-single-node milvus-p10 random-geo-radius-2048-angular-no-filters
./run_engine_test.sh milvus-single-node milvus-p10 arxiv-titles-384-angular-no-filters
./run_engine_test.sh milvus-single-node milvus-p10 deep-image-96-angular
```

 我发现后面的4个数据集都下不下来，所以我手动下载

![image-20251204211402658](./assets/image-20251204211402658.png)

下载之后要从本地电脑上传到服务器，但是上传的时候没有进度条，有一个特别简单的方法，就是ctrl+alt+insert打开任务管理器，然后看性能，就可以看到WLAN，可以看到一边下载，一边上传，速度拉满，说明任务没有断，不用慌。

如果网页下载完成，接收速率会归零。如果上传服务器完成，发送速率会归零。因为网络是“**全双工**的”，可以一边下载一边上传。

![image-20251204211920156](./assets/image-20251204211920156.png)

之前的glove-100-angular数据集是hdf5格式的，所以可以直接运行。

#### deep-image

对于deep-image，我们下载到的数据集文件叫做`deep-image-96-angular.hdf5`，我们需要在datasets当中新建一个文件夹，叫做`deep-image-96-angular`，然后把`deep-image-96-angular.hdf5`放进去。

构成这种格式：

```shell
(torch) dyx@server9050:~/VDTuner/vector-db-benchmark-master/datasets/deep-image-96-angular$ ls
deep-image-96-angular.hdf5
```

![image-20251204231636861](./assets/image-20251204231636861.png)

对比一下，会发现我们的路径正好拼接成了datasets.json当中所要求的路径，也就是：

```json
  {
    "name": "deep-image-96-angular",
    "path": "deep-image-96-angular/deep-image-96-angular.hdf5",
  },
```

这解释了手动添加的数据集可以被运行。

因为deep-image-96-angular也是hdf5格式的，所以也可以直接运行，可以发现这个数据集运行比起glove耗时很多，具体来说，大概花了10分钟，相比之下，glove只花了2分钟。

![image-20251204230319738](./assets/image-20251204230319738.png)

#### 其他3个数据集

其他三个数据集不是`.hdf5`格式的文件，而是一个压缩包，这个压缩包解压之后有很多文件，类似于random-100，而稍微观察一下datasets.json中random-100的结构，会发现它的path正好对应datasets当中的random-100

![image-20251205174822635](./assets/image-20251205174822635.png)

所以，我们知道了，对于这种解压之后有很多文件的数据集，datasets.json中的path指明了文件路径

#### random-match-keyword-100-angular-no-filters

观察datasets.json当中的path，可以发现

```json
    {
    "name": "random-match-keyword-100-angular-no-filters",
    "path": "random-match-keyword-100-angular/random_keywords_1m_no_filters",
    "link": "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_keywords_1m_no_filters.tgz"
  },
```

我们应该在datasets文件夹下面创建一个`random-match-keyword-100-angular/random_keywords_1m_no_filters`文件夹（文件夹套文件夹）

![image-20251205181440993](./assets/image-20251205181440993.png)

然后把`random_keywords_1m_no_filters.tgz`放进去，并且解压。

```shell
(torch) dyx@server9050:~/VDTuner/vector-db-benchmark-master/datasets/random-match-keyword-100-angular/random_keywords_1m_no_filters$ tar -zxvf random_keywords_1m_no_filters.tgz
vectors.npy
tests.jsonl
```

测试发现跑通了

![image-20251205181117064](./assets/image-20251205181117064.png)

#### random-geo-radius-2048-angular-no-filters

观察datasets.json当中的path，可以发现

```json
    {
    "name": "random-geo-radius-2048-angular-no-filters",
    "path": "random-geo-radius-2048-angular/random_geo_100k_no_filters",
    "link": "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_geo_100k_no_filters.tgz"
  },
```

我们应该在datasets文件夹下面创建一个`random-geo-radius-2048-angular/random_geo_100k_no_filters`文件夹（文件夹套文件夹）

![image-20251205181743525](./assets/image-20251205181743525.png)

然后把`random_geo_100k_no_filters.tgz`放进去，并且解压。

```shell
(torch) dyx@server9050:~/VDTuner/vector-db-benchmark-master/datasets/random-geo-radius-2048-angular/random_geo_100k_no_filters$ tar -zxvf random_geo_100k_no_filters.tgz
vectors.npy
tests.jsonl
```

测试发现跑通了

![image-20251205182318016](./assets/image-20251205182318016.png)

#### arxiv-titles

观察datasets.json当中的path，可以发现

```json
    {
    "name": "arxiv-titles-384-angular-no-filters",
    "path": "arxiv-titles-384-angular/arxiv_no_filters",
    "link": "https://storage.googleapis.com/ann-filtered-benchmark/datasets/arxiv_no_filters.tar.gz"
  },
```

我们应该在datasets文件夹下面创建一个`arxiv-titles-384-angular/arxiv_no_filters`文件夹（文件夹套文件夹），然后把`arxiv_no_filters.tar.gz`放进去，并且解压.

![image-20251205175247796](./assets/image-20251205175247796.png)

运行解压指令：

```shell
(torch) dyx@server9050:~/VDTuner/vector-db-benchmark-master/datasets/arxiv-titles-384-angular/arxiv_no_filters$ tar -zxvf arxiv_no_filters.tar.gz
._tests.jsonl
tests.jsonl
._vectors.npy
vectors.npy
```

测试发现跑通了：

![image-20251205183359091](./assets/image-20251205183359091.png)

### 修改VDTuner的配置

下面，为了运行VDTuner，也就是auto-configure文件夹，首先需要修改文件路径

![image-20251205184005383](./assets/image-20251205184005383.png)

在VDTuner项目当中搜索`/ytn/milvusTuning/`替换为`/dyx/VDTuner/`，也就是自己的路径

替换之后，可以去搜索/ytn/

可以发现没有更多内容，说明替换成功

![image-20251205184119451](./assets/image-20251205184119451.png)

然后在auto-configure文件夹（也就是调优的本体项目），当中，把`run_engine.sh`替换为`run_engine_test.sh`，也就是我们自己配置的这个更好的脚本。

![image-20251205184354463](./assets/image-20251205184354463.png)

假设我们没有sudo权限，需要把auto-configure中的sudo去掉，在在auto-configure文件夹（也就是调优的本体项目），当中，找到sudo。

![image-20251205185500463](./assets/image-20251205185500463.png)

如果没有去掉`sudo timeout`当中的sudo，直接去执行`./main_tuner.py`，大概率会遇到这个错误：

```shell
  File "/home/dyx/VDTuner/auto-configure/vdtuner/utils.py", line 125, in get_state
    y1, y2 = min(self.Y1_record), min(self.Y2_record)
             ^^^^^^^^^^^^^^^^^^^
ValueError: min() iterable argument is empty
```

现在，需要准备去运行VDTuner的调优主程序，也就是`./main_tuner.py`。

然而，如果直接运行，会遇到一个报错：

```shell
ImportError: cannot import name 'fit_gpytorch_model' from 'botorch.fit' (/home/dyx/.local/lib/python3.12/site-packages/botorch/fit.py). Did you mean: 'fit_gpytorch_mll'?
```

解决方案是去到

```shell
/home/dyx/VDTuner/auto-configure/vdtuner/optimizer_pobo_sa.py
```

然后进行全局替换，将`fit_gpytorch_model`替换为`fit_gpytorch_mll`，总共替换2处。

![image-20251205184948513](./assets/image-20251205184948513.png)









