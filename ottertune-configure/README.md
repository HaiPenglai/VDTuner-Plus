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



