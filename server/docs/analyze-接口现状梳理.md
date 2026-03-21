# `POST /durians/analyze` 接口现状与调整方案

本文记录当前 `server` 中 `durians.controller.ts` 的 `analyze` 接口实际执行流程，并补充后续准备采用的调整方案，重点覆盖：

- AI 如何基于 crop 对单个榴莲打分
- AI 输出如何约束为 JSON，并由后端汇总后返回前端
- crop 图片为何不需要持久化
- 小程序如何按阶段及时展示“识别中 / AI 打分中 / 已完成”
- 明确当前不做的事情

## 1. 当前实现现状

入口位于 `server/src/modules/durians/durians.controller.ts` 的 `POST /durians/analyze`。

当前一次请求会按下面顺序执行：

1. 使用 `FileInterceptor('file')` 接收上传文件，并限制文件大小为 10MB。
2. 如果没有传 `file`，直接返回 `400 Bad Request`，错误信息为 `file is required`。
3. 记录上传文件日志，包括 MIME type、原始文件名、字节大小。
4. 调用 `UploadsService.storeUploadedFile()` 将原图保存到本地 `uploads/` 目录，得到：
   - `localPath`
   - `fileUrl`
5. 调用 `DuriansService.createAnalysisTask()`，将 `localPath` 和 `fileUrl` 作为分析输入传入。
6. `DuriansService` 创建任务记录，初始状态为 `PENDING`，随后立即更新为 `DETECTING`。
7. `DuriansService` 同步调用 `CvService.detectAndAnnotate()`，将图片发给 Python `cv-service`。
8. `CvService` 优先读取本地 `imagePath`，组装 `multipart/form-data` 请求，转发到 `cv-service` 的 `/detect-and-annotate`。
9. Python 服务返回：
   - 标注后的整图 `annotated_image_base64`
   - 每个榴莲的裁剪图 `crop_image_base64`
   - 检测框坐标 `bbox`
   - 置信度 `confidence`
   - 稳定标签 `label`
10. `CvService` 当前会把标注图和 crop 图都再次落盘，转换为可访问 URL。
11. `DuriansService` 再调用 `AiService.summarizeDurianContext()`，让大模型基于整图生成一句简短摘要。
12. 最后任务只会更新：
   - `annotatedImageUrl`
   - `rawResult`
   - `aiSummary`
   - `status = SCORING`
13. Controller 返回：
   - `taskId`
   - `status`

如果 `cv-service` 请求失败，则任务状态会被更新为 `FAILED`，并记录 `errorMessage`。

## 2. 当前“分析榴莲”的实际能力

严格来说，当前系统实现的是“榴莲检测与标号”，不是完整的“榴莲评分”。

### 2.1 Python 侧当前在做什么

Python 侧核心逻辑在 `cv-service/app/services/detector.py`，大致流程如下：

1. 读取图片并完成基础校验。
2. 使用 YOLO 模型做目标检测。
3. 只保留类别名等于 `durian` 的检测框。
4. 对 `/detect-and-annotate` 结果应用业务过滤：
   - 若存在 `confidence > 0.6` 的结果，只保留前 9 个。
   - 否则若存在 `confidence > 0.4` 的结果，只保留前 3 个。
   - 否则返回 `没有识别到榴莲`。
5. 对检测框按“从上到下、从左到右”的视觉顺序排序，并给出稳定标签：
   - `A`
   - `B`
   - `C`
   - ...
6. 根据检测框从原图裁出 crop。
7. 在原图上绘制序号框，生成标注图。

### 2.2 当前系统能稳定产出的结果

当前能稳定得到的是：

- 检测到几个榴莲
- 每个榴莲的大致位置
- 每个榴莲的置信度
- 每个榴莲的标签 `A/B/C...`
- 一张带序号的标注图
- 每个榴莲对应的 crop
- 一段基于整图的 AI 简短摘要

### 2.3 当前系统还不能回答的问题

当前还不能稳定回答：

- 哪个榴莲更值得买
- 哪个成熟度更合适
- 哪个存在裂口、病斑、风险
- 推荐优先选择哪个编号

原因是目前没有真正把每个 crop 的信息转成结构化评分，也没有把任务推进到完整的 `DONE` 结果闭环。

## 3. 需要调整的方向

后续方案应从“检测与标号”升级为“检测、标号、逐个评分、汇总推荐”，但不引入额外 worker，也不在 Python 里增加第二阶段分类模型。

这意味着新的主流程应该是：

1. 先让 Python 完成检测和标号。
2. 立即把带序号的标注图返回给前端或可供前端轮询获取。
3. 后端拿 Python 返回的 crop，逐个或批量交给 AI 做评分。
4. AI 必须返回严格 JSON。
5. 后端把 AI 的 JSON 结果和检测结果汇总，生成最终任务结果。
6. 前端按阶段展示状态，而不是一直停在“分析中”。

## 4. 建议采用的目标方案

### 4.1 总体思路

建议保留当前“用户上传图片 -> Nest 调 Python -> Python 返回检测结果”的主链路，但把后半段改成真正的“评分链路”：

1. Python 负责：
   - 检测榴莲
   - 给榴莲标号
   - 返回标注图
   - 返回 crop 的 base64
2. Nest 负责：
   - 保存原图
   - 保存标注图
   - 不保存 crop 图
   - 调 AI 读取每个 crop 并输出评分 JSON
   - 汇总为最终结构化结果
3. 小程序负责：
   - 先展示原图上传成功
   - 再展示“已识别出 A/B/C...”
   - 再展示“AI 正在打分”
   - 最后展示评分和推荐结果

### 4.2 为什么 crop 不需要存储

crop 图本质上只是 AI 评分阶段的中间产物，不是长期业务资产。

建议调整为：

- Python 仍然返回 `crop_image_base64`
- Nest 在内存中直接把 crop 传给 AI
- AI 评分完成后，crop 不写本地文件、不生成公开 URL、不入库

这样做的好处：

- 减少磁盘写入
- 减少 URL 管理成本
- 减少无意义的文件残留
- 避免把仅供模型消费的中间数据当成正式静态资源

真正需要落盘并给前端展示的图片，只有：

- 用户原图
- 带序号的标注图

## 5. AI 对 crop 打分的方案

### 5.1 目标

AI 的职责不是“凭空写一段评价”，而是针对每个已经编号的榴莲输出结构化评分结果。

输出至少要覆盖：

- `label`
- `score`
- `summary`
- `reasons`
- `risks`
- `buyPriority`

同时，后端需要从所有 item 中汇总出：

- `recommendedLabel`
- `overallSummary`

### 5.2 推荐的数据流

建议的数据流如下：

1. Python 返回检测结果：
   - `annotated_image_base64`
   - `items[]`
   - 每个 item 包含：
     - `label`
     - `bbox`
     - `confidence`
     - `crop_image_base64`
2. Nest 保存标注图，更新任务状态为“已完成识别，准备 AI 打分”。
3. Nest 调 AI 时，把以下上下文一起传入：
   - 原图 URL 或原图二进制
   - 标注图 URL 或标注图二进制
   - 当前 crop 图
   - 当前榴莲 label
   - 当前榴莲 bbox
   - 当前榴莲 detection confidence
4. AI 输出严格 JSON。
5. Nest 校验 JSON，写入 `analysis_task_items`。
6. Nest 根据所有 item 的评分汇总出 `recommendedLabel` 和整体摘要。
7. 任务状态更新为 `DONE`。

### 5.3 推荐的单个榴莲评分字段

建议单个 item 使用如下结构：

```json
{
  "label": "A",
  "score": 86,
  "summary": "外形完整，果刺较均匀，成熟度较适中，适合买。",
  "reasons": [
    "果形较饱满",
    "刺间距较自然",
    "表面未见明显大面积破损"
  ],
  "risks": [
    "仅凭单张图片无法确认内部干包或坏果"
  ],
  "buyPriority": 1
}
```

字段约束建议：

- `label`: 必须和检测标签一致
- `score`: `0-100` 整数
- `summary`: 1 句短总结，便于前端直接展示
- `reasons`: 2 到 4 条正向判断依据
- `risks`: 0 到 3 条风险提示
- `buyPriority`: 正整数，`1` 代表最推荐

### 5.4 后端最终汇总结果建议

后端最终返回给前端的任务结果建议分为两层：

1. 任务级别
2. 榴莲 item 级别

推荐结构如下：

```json
{
  "taskId": "xxx",
  "status": "DONE",
  "stage": "COMPLETED",
  "sourceImageUrl": "https://...",
  "annotatedImageUrl": "https://...",
  "recommendedLabel": "A",
  "overallSummary": "本次共识别出 3 个榴莲，A 综合表现最好，B 次之，C 风险较高。",
  "items": [
    {
      "label": "A",
      "bbox": { "x1": 10, "y1": 20, "x2": 100, "y2": 120 },
      "confidence": 0.92,
      "score": 86,
      "summary": "外形完整，成熟度较适中，适合买。",
      "reasons": ["果形较饱满", "刺分布较均匀"],
      "risks": ["仅凭图片无法判断内部状态"],
      "buyPriority": 1
    }
  ]
}
```

其中：

- `bbox`、`confidence` 来自 Python 检测
- `score`、`summary`、`reasons`、`risks`、`buyPriority` 来自 AI
- `recommendedLabel`、`overallSummary` 由后端汇总生成

### 5.5 推荐的 AI Prompt 设计

建议不要让 AI 自由发挥，而是给它一个严格角色和严格输出格式。

#### System Prompt 建议

```text
你是一个榴莲挑选助手。你需要根据单个榴莲的局部图片，对这个榴莲进行购买价值评分。

你的目标不是判断“绝对真相”，而是基于图片中可见特征，给出谨慎、可解释、保守的购买建议。

你必须遵守以下规则：
1. 只能依据图片中可见的外观信息做判断，不能臆测不可见的内部果肉状态。
2. 如果证据不足，要明确写入 risks，不要过度自信。
3. 输出必须是合法 JSON，不允许输出 Markdown，不允许输出解释性前言，不允许使用代码块。
4. score 必须是 0 到 100 的整数。
5. reasons 必须是字符串数组，给出 2 到 4 条。
6. risks 必须是字符串数组，给出 0 到 3 条。
7. summary 必须简短，适合直接展示给用户。
8. label 必须与输入 label 完全一致。
```

#### User Prompt 建议

```text
请分析编号为 {{label}} 的榴莲 crop 图片，并返回 JSON。

已知上下文：
- 榴莲编号：{{label}}
- 检测框：{{bbox_json}}
- 检测置信度：{{confidence}}
- 这是一张从整图中裁切出来的局部图，可能存在角度、遮挡、光照影响。

评分标准：
- 重点关注外观完整度、果形是否饱满、刺的状态是否自然、表面是否有明显破损、是否存在肉眼可见风险。
- 不要根据看不见的内部状态做确定性结论。
- 如果图像信息不足，请在 risks 中说明。

请严格按照以下 JSON 结构返回：
{
  "label": "{{label}}",
  "score": 0,
  "summary": "",
  "reasons": [],
  "risks": [],
  "buyPriority": 0
}
```

#### 汇总 Prompt 建议

在单个 item 打分完成后，还可以追加一次“汇总 Prompt”，让 AI 或后端生成全局总结。

如果由后端规则汇总，建议：

- 按 `score` 从高到低排序
- `buyPriority = 1` 的 label 作为 `recommendedLabel`
- `overallSummary` 用模板拼接

如果由 AI 汇总，输入应为所有 item 的结构化 JSON，而不是再次传图。

推荐汇总输出：

```json
{
  "recommendedLabel": "A",
  "overallSummary": "本次识别出 3 个榴莲，A 综合外观最好，建议优先考虑；B 表现中等；C 存在更明显风险。"
}
```

### 5.6 为什么要用 JSON 约束 AI 输出

必须让 AI 输出 JSON，而不是自由文本，原因很明确：

- 后端需要稳定写库
- 前端需要稳定渲染
- 便于对字段做校验、兜底和排序
- 能直接生成 `analysis_task_items`
- 能降低 prompt 漂移对接口结构的影响

后端必须对 AI 输出增加校验：

- JSON 解析失败则判定本次评分失败
- `label` 不匹配则判定结果无效
- `score` 超出范围则拒收
- `reasons`、`risks` 类型错误则拒收

## 6. 前端状态展示方案

从用户视角，最重要的不是只看到一个“转圈”，而是知道现在到底走到哪一步。

建议任务状态区分为“数据库状态”和“前端展示阶段”两层。

### 6.1 建议的阶段定义

建议前端展示以下阶段：

1. `UPLOADING`
   - 用户刚提交图片
   - 小程序展示“图片上传中”
2. `DETECTING`
   - 服务端正在调用 Python 识别榴莲
   - 小程序展示“正在识别榴莲并编号”
3. `DETECTION_READY`
   - Python 已返回标注图
   - 小程序立即展示带 `A/B/C...` 序号的标注图
   - 同时提示“已识别榴莲，AI 正在逐个打分”
4. `SCORING`
   - 后端正在调用 AI 对 crop 打分
   - 小程序展示“AI 正在分析每个榴莲”
5. `DONE`
   - 所有评分和推荐都已生成
   - 小程序展示最终结果页
6. `FAILED`
   - 任一关键步骤失败
   - 小程序展示失败原因和重试入口

### 6.2 为什么要先返回标注图

用户一旦能看到序号图，就能理解“系统已经看懂了图里有哪些榴莲”，这会显著降低等待焦虑。

因此建议：

- Python 一完成检测和标号，就尽快让前端拿到 `annotatedImageUrl`
- 即便 AI 评分还没结束，前端也应该能先展示序号图

这样用户感知是：

1. 图片上传成功
2. 榴莲已识别并编号
3. AI 正在对 A/B/C 分别打分
4. 最终推荐结果已生成

### 6.3 返回结构建议

为了支撑阶段性展示，建议 `GET /durians/tasks/:taskId` 返回更完整的任务快照，例如：

```json
{
  "taskId": "xxx",
  "status": "SCORING",
  "stage": "SCORING",
  "sourceImageUrl": "https://...",
  "annotatedImageUrl": "https://...",
  "detectedLabels": ["A", "B", "C"],
  "message": "已完成榴莲识别，AI 正在逐个评分"
}
```

最终 `GET /durians/tasks/:taskId/result` 再返回完整结果：

```json
{
  "taskId": "xxx",
  "status": "DONE",
  "stage": "COMPLETED",
  "annotatedImageUrl": "https://...",
  "recommendedLabel": "A",
  "overallSummary": "A 综合表现最好",
  "items": []
}
```

## 7. 当前实现要删掉或调整的点

### 7.1 不再存储 crop 图片

当前 `CvService` 会把每个 crop 也保存为 URL。后续应去掉这一步。

目标改为：

- 标注图保留并落盘
- crop 只在当前请求或当前任务处理中存在
- crop 仅作为发给 AI 的中间输入

### 7.2 不再只做整图摘要

当前 AI 仅对整图生成一句摘要，价值有限。

后续重点应改为：

- 逐个 crop 的结构化评分
- 最终任务级别的汇总推荐

整图摘要如果保留，应该降级为附加信息，而不是主结果。

## 8. 明确不做的事情

以下方向目前不纳入方案：

1. 不引入 worker 或独立后台任务系统。
2. 不引入 Python 对每个 crop 做二阶段质量分类。
3. 不增加多任务视觉模型去判断成熟度、病斑、裂口等细分类别。

当前阶段只采用：

- Python 做目标检测和标号
- AI 基于 crop 做结构化评分
- Nest 汇总结果并返回前端

## 9. 一句话总结

当前 `POST /durians/analyze` 已经打通了“上传图片 -> Python 检测 -> 生成标注图”的基础链路；接下来应把方案收敛为“保留标注图、丢弃 crop 持久化、让 AI 基于 crop 输出严格 JSON、后端汇总为结构化评分结果，并让小程序按识别中 / AI 打分中 / 已完成三个关键阶段及时展示”。
