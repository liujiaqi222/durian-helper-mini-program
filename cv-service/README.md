# Python YOLO 榴莲检测微服务

这个目录是榴莲项目的 Python CV 微服务，负责把一张图片转换成主后端和多模态模型可用的检测结果。

当前已经实现的能力：

1. 加载 YOLO 模型并检测图片中的榴莲
2. 对检测结果按业务规则稳定排序并分配 `A/B/C...` 编号
3. 生成带编号的整图标注结果
4. 生成每个榴莲的局部裁剪图
5. 支持文件上传和 `image_url` 两种输入方式
6. 对输入图片做格式、大小、宽度校验
7. 提供基础测试用例，覆盖健康检查、检测流程和排序逻辑

它目前**不负责**：

- 评分哪个榴莲最好
- 生成购买建议文案
- 做多模型编排
- 持久化图片到对象存储

这层服务的职责很明确：解决“图里有哪些榴莲、每个榴莲在哪里、如何稳定编号、如何给下游提供整图和裁剪图”。

## 1. 微服务实现了什么

你可以把这个服务理解成一个专门做“榴莲目标检测预处理”的模块。它接到一张图片后，会完成下面这条链路：

1. 校验输入图片是否合法
2. 使用 YOLO 检测所有 `durian` 目标
3. 按检测框中心点做分层排序
4. 为每个榴莲分配稳定编号 `A/B/C...`
5. 根据检测框裁剪出每个榴莲的小图
6. 在原图上绘制框和编号标签
7. 返回结构化 JSON 结果给调用方

这样主后端后面只需要接：

1. 标注整图
2. 每个榴莲的 crop 图
3. 每个榴莲的稳定标签和 bbox

就可以继续做 LLM 评分、结果组装和前端展示。

## 2. 当前接口

### `GET /health`

健康检查接口，返回：

```json
{"status": "ok"}
```

### `POST /detect`

只返回检测结果，不返回标注图和裁剪图。

支持两种输入方式，二选一：

- `multipart/form-data` 上传 `file`
- `multipart/form-data` 传 `image_url`

返回示例：

```json
{
  "count": 2,
  "items": [
    {
      "label": "A",
      "class_name": "durian",
      "confidence": 0.9523,
      "bbox": {
        "x1": 40,
        "y1": 40,
        "x2": 220,
        "y2": 220
      }
    },
    {
      "label": "B",
      "class_name": "durian",
      "confidence": 0.9132,
      "bbox": {
        "x1": 260,
        "y1": 55,
        "x2": 430,
        "y2": 235
      }
    }
  ]
}
```

### `POST /detect-and-annotate`

返回检测结果、标注整图和每个榴莲的裁剪图。适合直接给主后端或 LLM 编排层调用。

返回字段在 `/detect` 基础上额外增加：

- `annotated_image_base64`
- `items[].crop_image_base64`
- 当没有满足条件的榴莲时，返回 `message`

这个接口额外带有一层业务筛选规则：

1. 如果检测结果里存在置信度 `> 0.7` 的榴莲，则返回这些榴莲
2. 如果不存在置信度 `> 0.7` 的榴莲，则返回：

```json
{
  "count": 0,
  "items": [],
  "message": "没有识别到榴莲"
}
```

### `GET /model-info`

返回当前运行中的模型和输入约束信息，例如：

- 模型路径
- 目标类别名
- 置信度阈值
- 支持的图片格式
- 最大上传大小
- 最小图片宽度

## 3. 编号为什么是稳定的

YOLO 原始输出顺序本身不可靠，如果直接把模型输出返回给主后端，同一张图多次调用时顺序可能漂移。这个服务已经在 Python 层做了稳定编号，规则是：

1. 先按检测框中心点 `y` 坐标分层
2. 同层按 `x` 坐标升序
3. 层间按 `y` 坐标升序
4. 最终分配 `A/B/C...` 标签

这意味着只要检测结果本身没有明显变化，返回的标签顺序就是可预期的。这个稳定性是后续 LLM 评分映射的前提。

## 4. 标注图和裁剪图怎么用

`/detect-and-annotate` 的输出可以直接喂给下游：

- `annotated_image_base64`
  - 用于给大模型看“整张图里 A、B、C 分别是谁”
- `crop_image_base64`
  - 用于让大模型看单个榴莲的细节，例如裂口、刺、形状、成熟度

推荐主后端调用流程：

1. 上传用户原图到 cv-service
2. 调用 `/detect-and-annotate`
3. 取回编号列表、整图标注和 crop 图
4. 把这些材料组织成多模态模型输入
5. 输出最终评分和推荐结果

如果由 NestJS 主后端统一接收用户上传，推荐改成：

1. 小程序把图片上传到主后端
2. 主后端把原图保存到本地 `uploads/` 目录
3. 主后端调用 `/detect-and-annotate` 时优先用 multipart `file` 方式直接上传本地文件
4. 只有在本地文件不可用时，才回退到 `image_url`

这样可以避免 cv-service 再发起一次 HTTP 下载，也更适合本地联调时共用同一台机器上的图片文件。

## 5. 输入校验规则

当前服务会拦截以下非法输入：

- 空文件
- 非图片文件
- 非 `jpg/jpeg/png/webp` 格式
- 大于 10MB 的图片
- 宽度小于 720px 的图片
- 同时传 `file` 和 `image_url`
- `file` 和 `image_url` 都不传

这样做是为了避免 YOLO 推理前就把明显不合规的数据带进来。

## 6. 目录说明

```text
cv-service/
├── README.md
├── requirements.txt
├── app
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   └── services
│       ├── detector.py
│       └── validators.py
└── tests
    ├── conftest.py
    ├── test_detect.py
    ├── test_health.py
    └── test_sorting.py
```

主要文件职责：

- `app/main.py`
  - FastAPI 入口与路由定义
- `app/services/detector.py`
  - YOLO 推理、排序编号、裁剪图、标注图生成
- `app/services/validators.py`
  - 图片输入校验
- `app/schemas.py`
  - 接口响应结构定义
- `tests/`
  - 接口和核心逻辑测试

## 7. 依赖安装与启动

建议使用 Python 3.12。

```bash
cd cv-service
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8010
```

启动后可访问：

- `http://127.0.0.1:8010/docs`
- `http://127.0.0.1:8010/health`

## 8. 模型与默认配置

当前默认配置：

- 模型路径：`models/durian-best.pt`
- 目标类别：`durian`
- 置信度阈值：`0.35`

仓库当前带的是 `models/best.pt`。如果 `durian-best.pt` 不存在，需要建立软链接或直接重命名为 `durian-best.pt`。

## 9. 测试

当前已经补了基础测试，覆盖：

- `/health`
- `/detect`
- `/detect-and-annotate`
- 输入校验
- 排序与稳定编号

运行方式：

```bash
cd cv-service
./.venv/bin/pytest tests -q
```

## 10. 下一步建议

这个微服务现在已经能提供 MVP 所需的核心 CV 材料。后续更值得继续做的是：

1. 把 base64 输出切换成对象存储 URL，避免响应体过大
2. 增加日志、耗时统计和错误监控
3. 补更多真实图片测试，特别是多排、多目标和空结果场景
4. 为 URL 下载增加白名单、超时和更严格的安全控制
5. 根据联调结果优化排序分层阈值
