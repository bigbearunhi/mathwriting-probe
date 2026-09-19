# MathWriting 轨迹识别可行性试验

**后续 CTC 试训已启动**：官方全量包已下载并通过 MD5 校验，当前进度见 `runs/ctc-full/pipeline.json`，模型和配置差异见 [CTC_README.md](CTC_README.md)。以下内容记录最初的小型 Seq2Seq 试验。

这是一次实验性训练链路检查，不是已经可用的公式识别模型。没有修改项目 `sources/`。

## 已获取的官方资源

- 仓库：`vendor/google-research/`，仅 sparse checkout `mathwriting/`。
- 上游：https://github.com/google-research/google-research/tree/master/mathwriting
- 固定提交及压缩包 SHA256：见 `provenance.json`。
- 数据：官方 `mathwriting-2024-excerpt.tgz`，约 1.5 MB。
- 解压后 train、valid、test、synthetic、symbols 各 100 条。
- 此试验仅用 train 训练、valid 验证；没有使用 test 进行训练或调参。
- **首次试验当时未下载约 2.9 GB 的全量数据；后续 CTC 阶段已下载。** 上游主要提供数据和读取/评估 notebook，不包含可直接运行的完整训练工程。

数据许可为 CC BY-NC-SA 4.0，不能据此默认用于商业产品；上游代码为 Apache-2.0。详细许可见官方 README。

## 数据处理

每条 InkML 的 trace 保存一笔的 `(x, y, time_ms)`，`normalizedLabel` 是目标 LaTeX。

`probe.py` 把原始数据导出为 `runs/<run>/train.jsonl` 和 `valid.jsonl`。每条记录含 id、latex、points；每个 point 含 x、y、time_ms、pen_up。**pen_up=1 表示该点是本笔最后一个点**，不是新笔第一个点。接入其他采集系统前需核对这一定义。

模型输入只用 x、y、pen_up，保留原始时间而暂不使用时间特征。坐标按整个公式等比例缩放并居中，保留宽高比。超过 384 点时逐笔抽样，保留各笔首尾点；不会简单截掉公式后半段，也不会跨笔插值。点数不足以保留所有笔端点时显式报错。此采样是快速试验策略，不代表最佳方案。

LaTeX 分词遵循官方示例的命令/字符规则；词表只从 train 构建，有 PAD/BOS/EOS/UNK。小样本词表对 valid 存在未见 token，数量写入 report.json；完整训练需重新建立足够覆盖的训练词表。

## 模型与训练

- 轨迹线性投影 `3 -> 128` + 正弦位置编码。
- 2 层 Transformer 编码器，2 层解码器，4 个头，FFN=512。
- LaTeX Embedding、因果自注意力、交叉注意力、词表输出层。
- 约 97 万参数，Adam，lr=0.001，梯度裁剪=1，batch=8。
- 训练用右移目标和 teacher forcing；PAD 不计入损失。
- 为便于过拟合检查，dropout=0，未用数据增强；这不是正式训练配置。
- 贪心解码以 EOS 停止，上限 128 token；编码器输出复用，未实现解码 KV cache。

当前环境：Python 3.10、PyTorch 2.9.0+cu128，RTX 4080 Laptop GPU，12 GB 显存。无需新装依赖。

## 重现

在此目录执行：

```bash
python3 -m unittest test_probe.py
python3 probe.py --steps 400 --train-limit 16 --out runs/smoke
python3 probe.py --steps 1000 --train-limit 0 --out runs/excerpt100
```

参数 `--train-limit 16` 会特意选择 16 条最短训练表达式做拟合检查；`0` 使用整个 train 目录。这是 minibatch 更新步数，不是 epoch 数。运行同名输出目录会覆盖该次试验产物；保留旧记录时使用新的 `--out`。

各 run 中包含 `checkpoint.pt`、`vocab.json`、训练日志、原始轨迹 JSONL 和 `report.json`。checkpoint 保存模型配置；脚本实际重载检查点并验证同一输入的生成结果一致。

## 实测结果及边界

1. 16 条训练样本、400 步：训练交叉熵 5.4078 -> 0.00173；自由生成完整匹配 16/16。这仅证明训练与生成链路能拟合数据，不代表泛化能力。
2. 100 条训练样本、1000 步：训练交叉熵 5.4529 -> 0.3414；teacher-forced token accuracy 89.53%。独立 valid 的 teacher-forced token accuracy 为 29.06%，loss 为 5.3084。
3. teacher-forced token accuracy 使用真实历史 token，**不能当作整条公式识别率**。报告中的验证自由生成只抽查 8 条，也不能当作正式 benchmark。
4. 两个测试验证了端点/抬笔语义、等比例归一化，以及修改未来目标或填充区域不影响前面预测。训练过程中损失和梯度均检查为有限值。
5. 当前设备足以开展小型轨迹 Transformer 实验。全量训练速度与显存不能从这次短序列、384 点预算的小样本试验直接外推。

## 下一阶段

下载官方全量包后，继续沿用官方 train/valid/test 划分；先统计点数、笔画数、LaTeX 长度和词表覆盖。当前脚本会把数据全部加载到内存，不适合直接把全量包交给它：应先改成缓存预处理 + 按需加载 + 按长度分桶，控制超长轨迹的注意力开销。

正式实验需在完整 valid 上自由生成，计算表达式完整匹配率和官方 token 编辑距离指标，再根据结果选择模型大小、点采样策略及时间特征。最后用真实采集设备的数据验证与适配。
