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

## 2. 常见 Bug 与经验总结

在运行 OtterTune 的 GPRGD 模型时，最常见的问题是 **NaN (Not a Number)** 错误和断言失败。以下是复现过程中总结的经验：

### Bug 1: 梯度下降过程中出现 NaN
**现象：** 训练模型时报错 `Tensor had NaN values` 或在 `predict` 阶段断言失败 `assert np.all(np.isfinite(yhats_it))`。
**原因：** 
1. **数据未缩放**：原始数据范围过大导致计算指数项时溢出。
2. **学习率过高**：Adam 优化器在迭代中步长过大，导致梯度爆炸。
3. **矩阵奇异性**：在计算核矩阵逆矩阵时，若点过于接近，会导致矩阵不可逆。

**解决方法：**
- **数据缩放**：务必将输入 `X`（配置参数）和输出 `y`（性能指标）通过 Min-Max 缩放至 `[0, 1]` 区间。
- **调整超参数**：
    - `length_scale`: 适当增大（如 1.0），使高斯核更平滑。
    - `ridge`: 显式设置正则化项（如 0.1），增加核矩阵的数值稳定性。
    - `learning_rate`: 降低学习率（从默认的 0.01 降至 0.001 或更低）。
- **初始化避让**：在 `predict` 阶段初始化预测点时，避开已有的训练数据点，防止 `sqrt(0)` 导致的梯度未定义问题。

---

## 3. 最小化测试用例运行

### 核心文件结构说明

- `ottertune/server/analysis/gp_tf.py`: OtterTune 的核心算法实现，包含 `GPRGD` 类（基于 TensorFlow 的高斯过程回归）。
- `ottertune-guide/simple_ottertune.py`: 经过优化的最小化运行示例，解决了数值稳定性问题。
- `ottertune-guide/README.md`: 针对最小化案例的详细教学说明。

为了验证环境和算法逻辑，我们构造了一个简单的优化案例：寻找函数 $y = (x - 5)^2$ 在 $[0, 10]$ 范围内的最小值。

### 运行步骤
```bash
# 进入引导目录
cd /home/dyx/VDTuner/ottertune-configure/ottertune-guide

# 运行最小化实现脚本
python simple_ottertune.py
```

### 预期输出
脚本将输出模拟的训练数据、模型训练过程（Debug 日志），以及最终推荐的配置点。
- **正确结果**：推荐点应接近 $x = 5$（缩放后约为 $0.5$）。
- **数值稳定**：不应出现 `AssertionError` 或 `NaN` 警告。

---

## 4. 将 OtterTune 应用于 VDTuner

我们实现了一个集成脚本 `main_ottertune.py`，它将 OtterTune 的 `GPRGD` 算法应用于 VDTuner 的参数调优任务。

### 核心功能
- **自动空间定义**：从 `whole_param.json` 自动读取 Milvus 的索引参数和系统参数。
- **初始采样**：使用拉丁超立方采样（LHS）生成初始实验数据。
- **单目标优化**：将多目标指标（RPS、Precision）组合为单一得分（当 Precision >= 0.9 时最小化 -RPS，否则视为无效配置）。
- **迭代推荐**：利用 OtterTune 的高斯过程模型不断预测并测试性能更好的配置。

### 运行方式

1. **激活环境**：
   ```bash
   conda activate ottertune
   ```

2. **启动调优**：
   ```bash
   cd /home/dyx/VDTuner/ottertune-configure
   python main_ottertune.py
   ```

### 关键代码说明 (`main_ottertune.py`)
- **路径配置**：脚本自动将 `ottertune/server` 和 VDTuner 的 `auto-configure` 目录加入 `sys.path`。
- **数据缩放**：在传递给 `GPRGD` 之前，所有配置参数和性能指标都会被缩放到 `[0, 1]` 区间，以保证模型的数值稳定性。
- **推荐逻辑**：
  - 使用 `LHS_sample` 进行冷启动。
  - 在每轮迭代中，使用 `model.fit` 训练模型。
  - 通过 `model.predict` 执行梯度下降，寻找能使收敛函数（Acquisition Function）最小化的新配置。




