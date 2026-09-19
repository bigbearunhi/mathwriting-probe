# Google MathWriting CTC 路线试训

## 与论文的关系

这是根据公开描述实现的 **CTC Transformer 训练试验**，不是 Google 官方训练源码或完整复现。官方 MathWriting 仓库仅提供数据、读取与评估示例。

来源：

- MathWriting §5.2：https://arxiv.org/html/2404.10690v2#S5.SS2
- 引用模型的 Appendix D：https://arxiv.org/html/2402.15307v1#A4
- Bézier 输入特征 §2.1.2：https://doi.org/10.1007/s10032-020-00350-4

对齐的配置：11 层编码器、模型维度 512、8 头、Swish/SiLU、dropout=0.15、CTC、Adam lr=0.001、真人 train + synthetic。用 microbatch=8、累积 32 次实现有效 batch=256。没有自回归解码器，也没有外接语言模型。

公开描述不足或不一致的部分明确做了本地选择：

1. 引用表写 units/head=256，但按标准注意力计算，仅注意力参数就超过报告的 35M。这里用 head_dim=64、FFN=2048，约 34.8M 参数；**并非声称表中的 256 就是 64**。
2. 使用正弦位置编码、pre-LN + 最终 LayerNorm、梯度裁剪范数 1；这些细节没有被原文完全指定。最初 post-LN 试跑中，100 步后输入梯度范数约 2e-18、验证输出退化成全 blank；这个失败的试跑单独保留，没有当成识别成功。
3. Bézier 采用端点约束、弦长参数化的三次最小二乘拟合，误差阈值 .01 和弧长/弦长约束递归分段。没有完全重现 Newton 参数重估、曲线重新合并或原作者未公开阈值。
4. 每条曲线提供 10 维特征：端点位移、控制点相对距离、角度、时间多项式系数和落笔标志。插入 pen-up 连接段保留笔画相对位置。等比例归一化；时间在每笔中按路径长度缩放。
5. 固定重复每个曲线特征两次以增加 CTC 对齐位置，**不根据真实标签改变输入**。超过 512 帧或不足以对齐标签的样本会统计并排除；验证指标对应保留子集，不能直接对比官方完整 benchmark。
6. 使用 BF16 前向、FP32 CTC 与 Adam，降低显存；本次计划先跑 1000 个 optimizer updates，**不是原文完整的 100,000 步**。

## 文件与运行

- `ctc_data.py`：流式读取 tgz，仅转换 train、synthetic、valid 到 SQLite；测试集不参与这次实验。
- `ctc_train.py`：训练、贪心 CTC 解码、token 编辑错误率、整式 exact match、可续训 checkpoint。
- `test_ctc.py`：重复 token/blank 规则、时间/笔画特征、坐标缩放不变性、padding 隔离、有限 CTC 梯度。
- `run_ctc_pipeline.py`：先校验官方全量包大小和 MD5，预处理，再自动启动有界训练。普通本地后台进程，不依赖定时任务。

```bash
python3 -m unittest test_ctc.py
python3 run_ctc_pipeline.py --steps 1000
```

最初 post-LN 的样本包训练与显存检查在 `runs/ctc-profile/`，稳定性调整后的检查在 `runs/ctc-prenorm/`，**它们只用于管线调试，不代表识别效果**。

全量试验目录 `runs/ctc-full/`：

- `pipeline.json`：当前阶段（等待下载、校验、预处理、训练、完成或失败）。
- `preparation.log` / `training.log`：各阶段日志。
- `config.json`：实际参数、词表、训练/验证过滤数。
- `metrics.jsonl`：逐更新步损失、时间、显存。
- `validation_*.json`：固定验证子集的 CTC loss、token_error_rate、exact_match、blank_fraction 和预测示例。
- `latest.pt`：模型、Adam 状态、随机数状态，支持同配置 `--resume`。
- `status.json`：最近一次已完成验证的训练进度。

当前后台任务由用户级 systemd 托管，服务名为 `mathwriting-download.service` 和 `mathwriting-ctc-trial.service`。下载使用官方 GCS JSON 媒体接口的分段请求，最终校验官方 MD5。

```bash
systemctl --user status mathwriting-download.service mathwriting-ctc-trial.service
# 如需停止全量训练及其子进程：
systemctl --user stop mathwriting-ctc-trial.service
```

训练的 `--steps` 是最终更新步编号；例如从 1000 步续到 2000 步需指定 `--steps 2000 --resume`。resume 加载的是本项目生成的可信 checkpoint。

## 评估与使用边界

词表仅从 train/synthetic 构建；valid 固定随机抽取最多 512 条可处理样本，未见 token 数也被报告。CTC token_error_rate 是编辑距离除以参考长度，插入很多时可以大于 100%；不等于 token classification accuracy。early blank collapse（预测全空白）会显式表现为 blank_fraction=1 和 token_error_rate=1，不算成功。

模型数据集尚未包含你的采集设备样本。任何论文指标都不是本地试验的承诺。数据是 CC BY-NC-SA 4.0，参见官方许可。原始文件保留；没有修改 `sources/`。

## 调整 Bézier 压缩程度

`ctc_data.py --tolerance` 控制分段拟合容差，默认 `0.01` 与既有缓存相同。较小（如 `0.005`）通常保留更多曲线段；较大（如 `0.02`）通常生成更少曲线段。它不是固定压缩倍数，段数不保证严格单调；弧长限制、笔画端点与抬笔连接仍然保留。误差计算包含归一化 x/y 与按笔画弧长缩放的时间，因此不能直接等同于纯二维形状误差百分比。

```bash
python3 ctc_data.py --archive data/mathwriting-2024.tgz \
  --out data/ctc-tolerance-002 --workers 8 --tolerance 0.02
```

程序拒绝覆盖已有缓存。改变容差需要生成新缓存，并重新检查输出长度与 CTC 对齐条件；不会自动更改既有模型或训练。`preparation.json` 记录数值及误差空间。Python 调用也支持 `curve_features(strokes, tolerance=0.02)` 和 `prepare(..., tolerance=0.02)`。

预处理 HTML 的“压缩强度”提供 `0.005 / 0.01 / 0.02 / 0.05` 四档。运行 `python3 build_preprocess_viewer.py` 会为抽样笔迹预计算各档位，并从 `viewer_templates/preprocess.html` 生成页面。网页切换无需 GPU/服务端，只做几何对比，原模型识别结果不会重新计算。每档会更新段数、CTC 帧数、长度条件和近似几何偏差；同一样本共用固定视窗。默认 `0.01` 对照训练缓存验证，其余档位不写入训练数据库。
