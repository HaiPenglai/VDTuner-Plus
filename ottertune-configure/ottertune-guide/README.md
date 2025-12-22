# OtterTune 最小化教学案例

本指南旨在通过一个简单的例子展示 OtterTune 的核心调优逻辑：**高斯过程回归 (Gaussian Process Regression, GPR)**。

OtterTune 的核心思想是利用已有的实验数据（配置和对应的性能指标）来训练一个机器学习模型（GPR），然后利用该模型预测并推荐更好的配置。

## 核心逻辑：GPR 推荐

在 OtterTune 中，推荐过程通常如下：
1. **数据准备**：收集一组已知的配置 `X` 和它们对应的性能 `y`（例如延迟或吞吐量）。
2. **训练模型**：使用 `(X, y)` 训练 GPR 模型。
3. **寻找最优**：在模型上执行梯度下降，寻找能使预测性能最优的配置。

## 运行示例

我们提供了一个简单的脚本 `simple_ottertune.py`，它尝试最小化函数 $y = (x - 5)^2$。

### 1. 环境准备

确保你已经创建并激活了 `ottertune` 环境：

```bash
conda activate ottertune
```

### 2. 运行脚本

```bash
# 进入引导目录
cd /home/dyx/VDTuner/ottertune-configure/ottertune-guide

# 运行脚本
python simple_ottertune.py
```

### 3. 代码解析

`simple_ottertune.py` 的核心部分使用了 OtterTune 源码中的 `GPRGD` 类。为了确保模型数值稳定性（避免 NaN），我们在示例中应用了以下技巧：

1. **数据缩放 (Scaling)**：将配置 `X` 和性能 `y` 都缩放到 `[0, 1]` 区间。
2. **超参数调整**：
   - `length_scale=1.0`: 增加长度尺度，使模型更平滑。
   - `ridge=0.1`: 增加正则化项，防止矩阵求逆时出现奇异矩阵。
   - `learning_rate=0.001`: 减小学习率，防止梯度爆炸。
3. **避开重复点**：初始化预测点（`X_test`）时避开已有的训练点，防止 `sqrt(0)` 导致的梯度计算错误。

```python
from analysis.gp_tf import GPRGD

# 初始化模型 (带稳定超参数)
model = GPRGD(length_scale=1.0, magnitude=1.0, ridge=0.1, max_iter=20, learning_rate=0.001)

# 拟合缩放后的数据
model.fit(X_train_scaled, y_train_scaled, X_min=np.array([0.0]), X_max=np.array([1.0]))

# 推荐新配置
results = model.predict(X_test_initial)
```

通过这个例子，你可以看到 OtterTune 是如何通过历史数据来“学习”性能曲线并寻找最优点的。这种机制是 OtterTune 自动调优数据库配置的核心。