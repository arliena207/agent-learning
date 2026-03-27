# Chain-of-Visual-Thought 论文阅读笔记

### 论文标题

**Chain-of-Visual-Thought (CoVT): Visual Thinking for Vision-Language Models**

### 论文链接

- arXiv 页面：https://arxiv.org/abs/2511.19418
- PDF 链接：https://arxiv.org/pdf/2511.19418

### 论文作者

Yiming Qin, Bomin Wei, Jiaxin Ge, Konstantinos Kallidromitis, Stephanie Fu, Trevor Darrell, XuDong Wang

## 1. 背景与 Limitation

现有 VLM 对语言推理已较强，但在 perception-heavy 任务（深度、边界、空间关系、细粒度识别）上仍有明显短板。核心原因是：

1. 跨空间维度的密集视觉信息难以被有效建模。  
2. 视觉信息压缩到离散 text token 后，连续空间线索（几何、边缘、布局）损失明显。  
3. Text-only CoT 在长链推理中更容易累积误差。  
4. 训练监督主要集中在文本输出，缺少对中间视觉思考过程的直接约束。

## 2. 核心工作（CoVT）

CoVT 的核心是让 VLM 不仅通过离散文本 token 推理，也通过连续视觉 token 推理。

- 文本 token（discrete）：负责语言逻辑与符号推理。  
- 视觉 token（continuous）：负责承载密集感知信息（深度/分割/边缘/语义特征）。

推理范式从：
`Image -> text tokens -> reasoning`
变为：
`Image -> visual tokens + text tokens -> joint reasoning`

训练时，模型自回归预测视觉 token，并用重建 + 蒸馏目标对齐稠密视觉监督；
推理时，模型可直接在连续视觉 token 空间进行思考，再输出文本答案。

## 3. 训练 Pipeline 总览

`图像输入 -> 生成/采样连续视觉token -> 投影到多视觉专家空间 -> 重建与特征对齐训练 -> 在<think>中联合文本+视觉token推理 -> 输出文本答案`

可以理解为两条并行链路：

- Prompt-level alignment：token -> decoder -> 重建 segmentation/depth/edge 等稠密信号。  
- Feature-level alignment：token -> projection -> 对齐语义特征空间（如 DINO）。

## 4. 四种能力 Token

CoVT 将视觉能力拆成四类，并由对应视觉专家信号监督：

1. Instance Recognition（实例识别）  
对应对象实例层面的可分辨性（ SAM 相关监督）。

2. Spatial Relationship（空间关系）  
对应几何深度与前后关系（ Depth Anything 监督）。

3. Structure Detection（结构检测）  
对应边缘/轮廓等结构信息（ PIDINet 监督）。

4. Semantic Information（语义信息）  
对应高层语义特征表达（ DINO 特征对齐）。

## 5. Training 流程

总体loss：
`L_total = L_text + L_visual`
其中 `L_visual` 是多任务视觉重建/对齐损失。

四阶段训练设计：

1. Comprehension  
`<image> -> visual tokens -> question -> answer`  
目标：先让模型理解 visual token 含义。

2. Generation  
`<image> -> generate visual tokens -> answer`  
目标：让模型学会稳定生成 visual token。

3. Reasoning  
`<image> -> <think> (visual tokens + reasoning) </think> -> answer`  
目标：把 visual token 融入推理链。

4. Efficient Reasoning  
训练时随机使用/丢弃部分视觉监督分支（如只用 depth、或不用 segmentation），提升推理阶段对视觉信息调用的灵活性与效率。

## 6. Outcome

实验显示 CoVT 在多个 perception-heavy benchmark 上带来约 **3% - 16%** 提升。文中报告在 CV-Bench 上有明显增益，在 depth 子任务上提升更突出。

除了指标提升，CoVT 还带来三点关键价值：

- 更 precise：视觉相关判断更准确。  
- 更 grounded：推理更依赖真实视觉证据，而非纯语言猜测。  
- 更 interpretable：连续视觉 token 可被解码，便于观察模型的“视觉思考过程”。

## 7. 总结
现有 VLM 将视觉压到语言空间推理，导致细粒度信息丢失。CoVT 通过引入连续 visual tokens，把视觉信息直接带入推理过程，并结合多维 token 设计与分阶段训练，使模型从 text-only 推理扩展到 visual+language 联合推理，在视觉任务上明显提升。

## 8. 我的思考
我觉得这篇工作的关键是它在尝试改变 VLM 的推理表示。很多 VLM 的问题在于看见之后只能翻译成语言再想，这一步本身就会丢掉很多信息。CoVT 让我感觉，未来多模态模型如果想把视觉推理做深，可能必须让视觉表征本身进入思考过程，而不是永远给语言做前处理。