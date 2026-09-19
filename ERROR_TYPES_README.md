# 识别错误分类

运行（在 mathwriting-probe 目录）：

```bash
python3 -m unittest test_error_types -q
python3 error_types.py --input runs/token-error-analysis/predictions.json --out runs/error-types --workers 4
```

输入：JSON 数组，每项包含 id、truth、prediction。分析已有预测，不重新训练，也不加载模型。当前默认示例预测来自第43500步、2048条验证样本。

输出：summary.json 汇总；details.json 含逐样本分类、编辑对齐和编译错误原因；details.csv 便于筛选。

分类可以重叠，不能把比例相加成100%：
- number_error：最小 Token 编辑涉及数字。例如1→l同时涉及数字和符号。
- symbol_error：涉及字母、希腊字母、运算符等，排除数字、空白以及单独列出的结构控制Token。
- syntax_error：输出的数学Token序列无法在受支持的LaTeX环境中编译；不是仅根据它与标准答案不同判定。
- structure_token_error：上下标、花括号、frac等结构Token有差异，不一定语法非法。
- whitespace_error：涉及空白Token的差异。CTC blank不包含在最终输出中。

语法检查用pdflatex + amsmath/amssymb/amsfonts，临时目录运行，禁用shell escape，限制输入为本地MathWriting词表，超时8秒。Token之间加入空格，保留数学Token边界，避免TeX将连续^^作为字符编码转义。未知命令、词表外内容、编译器缺失或超时标为unknown，不武断当作语法错误。标准答案也编译，单独统计标准答案异常，以及参考答案有效但预测非法的introduced_syntax_error。

限制：这是受限数学LaTeX检查器，不支持任意用户宏。能编译不等于数学含义正确；空字符串可编译但识别可能全漏。最小编辑对齐可能不唯一。数字统计是Token层面的识别差异，不是字符框/笔画级检测准确率。

## 符号细分

运行同一命令还会生成 symbol_errors.csv（按标准答案侧错误总数排序）、symbol_confusions.csv（具体混淆对）和 symbol_breakdown.json。

每种符号分别统计：标准答案出现次数、被替换次数、被删除次数、其他Token误识别为它的次数、额外插入次数。错误率=(被替换+被删除)/标准答案出现次数。后两类单独统计，不混入这个分母。没有标准答案出现记录时错误率为空。低频符号的百分比不稳定，应同时查看次数。这里分析Token编辑对齐，不是笔画级视觉分类；命令粘连可能改变Token边界。

## 原始笔迹与错误查看器

```bash
# 生成全部错误的可搜索网页
python3 inspect_error.py
# 只查看一个ID
python3 inspect_error.py --id e6eec86f43b20780 --input runs/long-formula-results.json
# 只筛选x被认成X（默认输入为当前分类结果）
python3 inspect_error.py --reference x --prediction X --out runs/x-confusions
```

输入可为含id/truth/prediction的JSON数组，也支持含records数组的报告。默认输入为runs/error-types/details.json。
生成runs/error-viewer/index.html，直接浏览器打开即可，无外部网络依赖。支持ID/LaTeX搜索、错误类型筛选、标准Token与错误Token配对筛选、逐项编辑查看，一次只显示1条，通过上一条/下一条切换。原始坐标保存在同目录source_samples.json，来源清单在manifest.json。网页的差异高亮采用序列匹配，逐项统计采用最小编辑距离，模糊对齐下高亮分段可能不同。

查看器现支持识别正确/识别错误/全部切换，默认显示错误。当前第43500步共2048条，正确922条，错误1126条。均一次只显示一条。
