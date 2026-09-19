# 换电脑继续训练（Bézier 0.02 + 笔内卷积）

代码及迁移附件位于私有仓库 bigbearunhi/mathwriting-probe。
Release 标签：`migration-bezier002-20260919`。附件的 `MANIFEST.json` 记录存档步数、文件大小和 SHA256。

## 1. 环境

建议 Linux、Python 3.10+、NVIDIA GPU。原环境 Python 3.10、PyTorch 2.9.0+cu128、NumPy 2.2.6。
新 GPU（包括 RTX 5090）需安装支持它的驱动及 CUDA 版 PyTorch；以下使用 CUDA 12.8 构建。

```bash
gh auth login
gh repo clone bigbearunhi/mathwriting-probe
cd mathwriting-probe
git checkout migration-bezier002-20260919
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.2.6
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())'
```

## 2. 下载并校验

```bash
mkdir -p migration-assets
gh release download migration-bezier002-20260919 -R bigbearunhi/mathwriting-probe -D migration-assets
(cd migration-assets && sha256sum -c SHA256SUMS)
mkdir -p data
cat migration-assets/raw-mathwriting.tgz.part-* > data/mathwriting-2024.tgz
cat migration-assets/cache-bezier002.tar.gz.part-* | tar -xzf -
tar -xzf migration-assets/training-snapshot.tar.gz
```

缓存是完整的当前 0.02 预处理缓存，原始 tgz 也已包含。无需下载旧 0.01 缓存、旧训练目录或重新预处理。原始数据包含测试集，测试集不用于训练。

## 3. 核对并续训

```bash
python resume_stroke_conv.py --checkpoint runs/migrated-strokeconv/latest.pt --out runs/resume-check --check-only
python resume_stroke_conv.py --checkpoint runs/migrated-strokeconv/latest.pt --out runs/strokeconv-next --steps 10000 --microbatch 16
```

`--steps 10000` 表示训练到**累计 10,000 步**，不是再加 10,000 步；可改为希望达到的总步数。
`--out` 必须是新目录。默认恢复原学习率 0.0001、Adam 状态、随机状态、相同验证样本；每 500 步保存和验证。
有效 batch 固定 256，microbatch 可按显存改为 32 或 64（须整除 256）。跨硬件或改变 microbatch 后不保证浮点结果逐位相同。
`--check-only` 已在原机用真实存档完成加载和 GPU 前向测试；CPU 小模型测试确认优化器和随机状态恢复后的下一步更新一致。

模型：10维 Bézier 特征 → 笔内两层卷积（10→64→512、核3、步长1、SiLU、残差）→ 重复2次 → 11层 Transformer encoder → CTC。
卷积不跨抬笔、不缩短序列。`best.pt` 是打包时已评估的最低 token 错误率存档，`latest.pt` 是打包时最新完整存档。两者都带优化器状态。

附件不包括历次所有旧模型、旧缓存；包含迁移当前算法所需的数据、当前最新及最佳存档、配置、验证结果与实际运行源码快照。
数据来源与许可见原始 MathWriting 归档及项目 provenance.json。请只加载自己信任的 PyTorch 存档。
