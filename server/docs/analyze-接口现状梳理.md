# `POST /durians/analyze` 接口现状梳理

本文记录当前 `server` 中 `durians.controller.ts` 的 `POST /durians/analyze` 实际实现方式，并同步说明当前 `cv-service`、`server`、`client` 三端已经完成的改造，重点覆盖：

- 当前上传、检测、AI 评分、结果汇总的真实链路
- AI 如何基于 crop 对单个榴莲输出结构化评分
- crop 图片为何不再持久化
- 小程序如何按阶段展示“识别中 / AI 打分中 / 已完成”
- 当前已实现与仍未实现的边界

## 1. 当前实现现状

入口位于 `server/src/modules/durians/durians.controller.ts` 的 `POST /durians/analyze`。

当前一次请求按下面顺序执行：

1. 使用 `FileInterceptor('file')` 接收上传文件，并限制文件大小为 10MB。
2. 如果没有传 `file`，直接返回 `400 Bad Request`，错误信息为 `file is required`。
3. 记录上传文件日志，包括 MIME type、原始文件名、字节大小。
4. 调用 `UploadsService.storeUploadedFile()` 将原图保存到本地 `uploads/` 目录，得到：
   - `localPath`
   - `fileUrl`
5. 调用 `DuriansService.createAnalysisTask()` 创建任务，写入：
   - `sourceImagePath`
   - `sourceImageUrl`
   - `status = PENDING`
6. Controller 立即返回：
   - `taskId`
   - `status`
7. `DuriansService` 在单进程内异步启动后台处理，不阻塞本次 HTTP 响应。
8. 后台任务先把状态推进到 `DETECTING`，再调用 `CvService.detectAndAnnotate()`。
9. `CvService` 优先读取本地 `imagePath`，组装 `multipart/form-data` 请求，转发到 Python `cv-service` 的 `/detect-and-annotate`。
10. Python 服务返回：
   - 标注后的整图 `annotated_image_base64`
   - 每个榴莲的裁剪图 `crop_image_base64`
   - 检测框坐标 `bbox`
   - 置信度 `confidence`
   - 稳定标签 `label`
11. `CvService` 只会把标注图落盘，生成 `annotatedImageUrl`；crop 不落盘、不生成 URL。
12. `DuriansService` 记录检测阶段结果，更新：
   - `annotatedImageUrl`
   - `detectedCount`
   - `detectedLabels`
   - `rawResult`
   - `status = SCORING`
13. `DuriansService` 调用 `AiService.scoreDurians()`，把每个榴莲的：
   - `label`
   - `bbox`
   - `confidence`
   - `cropImageBase64`
   - 原图信息
   交给 AI 评分。
14. `AiService` 要求模型返回严格 JSON；如果没有配置 AI key，则回退到基于检测置信度的启发式评分。
15. 后端把 AI 评分结果写入 `analysis_task_items`，再汇总：
   - `recommendedLabel`
   - `overallSummary`
   - `status = DONE`
16. 如果 CV 请求失败、AI 输出非法、或任务中途报错，则任务被更新为：
   - `status = FAILED`
   - `errorMessage = 失败原因`

## 2. 当前“分析榴莲”的实际能力

当前系统已经不再只是“榴莲检测与标号”，而是已经具备完整的 MVP 闭环：

- 检测榴莲
- 对榴莲稳定编号
- 生成带编号的标注图
- 提取每个榴莲 crop 供 AI 使用
- 对每个榴莲输出结构化评分
- 汇总推荐编号和整体结论
- 支持前端按阶段轮询任务状态

但这仍然是 MVP，评分结果仍然属于“基于图片可见信息的保守建议”，不是专业验果结论。

### 2.1 Python 侧当前在做什么

Python 侧核心逻辑在 `cv-service/app/services/detector.py`，当前职责保持不变：

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

Python 当前仍然只负责“检测、编号、裁剪、标注”，不负责评分，不负责推荐，也不负责持久化业务结果。

### 2.2 当前系统能稳定产出的结果

当前能稳定得到的是：

- 检测到几个榴莲
- 每个榴莲的大致位置 `bbox`
- 每个榴莲的置信度 `confidence`
- 每个榴莲的标签 `A/B/C...`
- 一张带序号的标注图
- 每个榴莲对应的 crop base64
- 每个榴莲的结构化评分：
  - `score`
  - `summary`
  - `reasons`
  - `risks`
  - `buyPriority`
- 任务级推荐结果：
  - `recommendedLabel`
  - `overallSummary`

### 2.3 当前仍然不能稳定回答的问题

当前还不能稳定回答：

- 榴莲内部是否干包、坏果、夹生
- 仅凭单图精确判断成熟度
- 作为专业采购标准的绝对“最优果”

原因很明确：当前评分仍然依赖单张外观图和通用 AI 判断，不能替代线下开果或专业经验。

## 3. 当前主流程

### 3.1 总体思路

当前已落地的主链路是：

1. 小程序上传原图到 `server`
2. `server` 保存原图并创建任务
3. 后台调用 `cv-service` 检测、编号、生成标注图和 crop
4. `server` 保存标注图，但不保存 crop
5. `server` 调 AI 对每个 crop 输出严格 JSON
6. `server` 汇总任务级推荐结果
7. 小程序轮询任务状态并按阶段展示

### 3.2 为什么 crop 不再存储

crop 图本质上只是 AI 评分阶段的中间产物，不是长期业务资产。

当前实现已经调整为：

- Python 仍然返回 `crop_image_base64`
- Nest 在内存中直接把 crop 传给 AI
- AI 评分完成后，crop 不写本地文件、不生成公开 URL、不入库

这样做的好处：

- 减少磁盘写入
- 减少 URL 管理成本
- 避免保留无意义中间文件
- 明确“可展示图片”和“仅供模型消费图片”的边界

当前真正需要落盘并给前端展示的图片，只有：

- 用户原图
- 带序号的标注图

## 4. AI 对 crop 打分的方案

### 4.1 当前目标

AI 的职责不是自由写评价，而是针对每个已编号榴莲输出结构化评分结果，并由后端再汇总成任务级推荐。

当前结构化评分至少覆盖：

- `label`
- `score`
- `summary`
- `reasons`
- `risks`
- `buyPriority`

任务级汇总至少覆盖：

- `recommendedLabel`
- `overallSummary`

### 4.2 当前数据流

当前实际数据流如下：

1. Python 返回检测结果：
   - `annotated_image_base64`
   - `count`
   - `items[]`
2. 每个 `item` 包含：
   - `label`
   - `bbox`
   - `confidence`
   - `crop_image_base64`
3. Nest 保存标注图，更新任务：
   - `annotatedImageUrl`
   - `detectedCount`
   - `detectedLabels`
   - `status = SCORING`
4. Nest 调 AI 时，把以下上下文传入：
   - 原图 URL 或原图二进制
   - 当前 crop 图
   - 当前榴莲 label
   - 当前榴莲 bbox
   - 当前榴莲 detection confidence
5. AI 输出严格 JSON。
6. 后端校验 JSON，并写入 `analysis_task_items`。
7. 后端根据所有 item 的评分汇总出 `recommendedLabel` 和 `overallSummary`。
8. 任务状态更新为 `DONE`。

### 4.3 当前单个榴莲评分字段

当前单个 item 的目标结构如下：

```json
{
  "label": "A",
  "score": 86,
  "summary": "外形完整，成熟度较适中，适合买。",
  "reasons": [
    "果形较饱满",
    "刺分布较均匀"
  ],
  "risks": [
    "仅凭图片无法判断内部状态"
  ],
  "buyPriority": 1
}
```

字段约束：

- `label`: 必须和检测标签一致
- `score`: `0-100` 整数
- `summary`: 1 句短总结
- `reasons`: 2 到 4 条正向依据
- `risks`: 0 到 3 条风险提示
- `buyPriority`: 正整数，`1` 表示最推荐

### 4.4 当前最终结果结构

当前 `GET /durians/tasks/:taskId/result` 返回的核心结构分为两层：

1. 任务级别
2. 榴莲 item 级别

返回示意如下：

```json
{
  "id": "xxx",
  "status": "DONE",
  "sourceImageUrl": "https://...",
  "annotatedImageUrl": "https://...",
  "recommendedLabel": "A",
  "overallSummary": "本次共识别出 3 个榴莲，A 综合表现最好，建议优先选择。",
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

### 4.5 当前 AI 输出约束

当前 AI 已经按“严格 JSON”思路实现，后端会做校验，不接受自由文本主结果。

后端当前会检查：

- JSON 能否被正常解析
- `label` 是否与检测标签一致
- `score` 是否在合法范围内
- `reasons`、`risks` 是否满足预期数组结构

如果校验失败，本次任务会进入 `FAILED`，并记录错误信息。

### 4.6 当前无 AI Key 时的兜底

如果没有配置 `ai.apiKey`，当前不会直接中断整条链路，而是回退到启发式评分：

- 分数基于 detection confidence 映射
- 推荐文案使用固定模板
- 风险提示使用保守默认文案

这样可以保证本地联调时仍然能走通 `DONE` 结果闭环。

## 5. 前端状态展示方案

从用户视角，最重要的不是看到一个“转圈”，而是知道当前处在哪个阶段。

当前前端已经按任务状态分阶段展示。

### 5.1 当前阶段定义

当前前端主要围绕以下状态工作：

1. `PENDING`
   - 任务已创建，后台尚未开始或刚开始处理
   - 小程序展示“任务已创建，准备开始分析”
2. `DETECTING`
   - 服务端正在调用 Python 识别榴莲
   - 小程序展示“正在识别榴莲位置和编号”
3. `SCORING`
   - Python 已返回标注图，后端正在调用 AI 逐个评分
   - 小程序会优先展示标注图和已识别的 `A/B/C...`
   - 同时提示“AI 正在生成评分和购买建议”
4. `DONE`
   - 所有评分和推荐都已生成
   - 小程序展示最终结果页
5. `FAILED`
   - 任一关键步骤失败
   - 小程序展示失败原因和重试入口

### 5.2 为什么要先返回标注图

只要用户先看到序号图，就能理解“系统已经识别出货架里的榴莲”。

因此当前实现中：

- 检测一完成，任务详情接口就会带上 `annotatedImageUrl`
- 即便 AI 评分还没结束，前端也能先展示标注图
- 结果页会在 `SCORING` 阶段提示“已识别 X 个榴莲：A、B、C”

### 5.3 当前接口返回结构

为了支撑阶段性展示，当前 `GET /durians/tasks/:taskId` 已返回更完整的任务快照，核心字段包括：

```json
{
  "id": "xxx",
  "status": "SCORING",
  "sourceImageUrl": "https://...",
  "annotatedImageUrl": "https://...",
  "detectedCount": 3,
  "detectedLabels": ["A", "B", "C"],
  "errorMessage": null,
  "overallSummary": null,
  "recommendedLabel": null
}
```

最终 `GET /durians/tasks/:taskId/result` 再返回完整结果：

```json
{
  "id": "xxx",
  "status": "DONE",
  "annotatedImageUrl": "https://...",
  "recommendedLabel": "A",
  "overallSummary": "A 综合表现最好",
  "items": []
}
```

## 6. 当前实现中已经完成的关键调整

### 6.1 已去掉 crop 持久化

当前 `CvService` 已不再把每个 crop 保存为 URL。

当前行为是：

- 标注图保留并落盘
- crop 只在当前任务处理中存在
- crop 仅作为发给 AI 的中间输入

### 6.2 已从整图摘要切到逐个评分

当前 AI 已不再只做整图一句摘要。

当前主结果已经切换为：

- 逐个 crop 的结构化评分
- 任务级别的推荐汇总

`overallSummary` 是当前主汇总字段，旧的 `aiSummary` 已不再作为主契约使用。

### 6.3 已增加任务级阶段字段

当前任务模型已经补齐：

- `detectedCount`
- `detectedLabels`
- `overallSummary`

item 模型已经补齐：

- `bbox`
- `confidence`
- `score`
- `summary`
- `reasons`
- `risks`
- `buyPriority`

## 7. 当前明确不做的事情

以下方向当前仍不纳入实现范围：

1. 不引入 worker 或独立后台任务系统。
2. 不引入 Python 对每个 crop 做二阶段质量分类。
3. 不增加多任务视觉模型去判断成熟度、病斑、裂口等细分类别。
4. 不对 crop 做对象存储或公开静态访问。
5. 前端不做历史记录、登录态、分享卡片。

当前阶段只采用：

- Python 做目标检测和标号
- AI 基于 crop 做结构化评分
- Nest 汇总结果并返回前端
- 小程序按阶段轮询并展示

## 8. 一句话总结

当前 `POST /durians/analyze` 已经完成从“同步检测半成品接口”到“异步任务编排接口”的改造：请求会快速返回任务 ID，后台完成 Python 检测、标注图生成、crop 驱动的 AI 结构化评分与最终推荐汇总，小程序则按 `PENDING / DETECTING / SCORING / DONE / FAILED` 阶段逐步展示识别与评分结果。
