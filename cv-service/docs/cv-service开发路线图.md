# cv-service 开发路线图

## 现状分析

cv-service 目前只完成了最基础的一步：**原始检测**。接收一张图片，用 YOLO 找出所有榴莲的边界框 + 置信度，返回 JSON。没有编号、没有标注图、没有裁剪图、没有测试用例。

根据 `server/docs/方案.md` 的 MVP 流程，检测服务在整条链路中承担的职责是：

> 检测 → 去重 → 排序 → 编号 → 生成标注图 → 生成裁剪图

当前只完成了第一步。

已完成的部分：

- FastAPI 服务骨架（`app/main.py`）
- YOLO 模型加载与推理（`app/services/detector.py`）
- 请求/响应 schema 定义（`app/schemas.py`）
- 配置管理（`app/config.py`）
- `/health` 健康检查接口
- `/detect` 检测接口（返回原始边界框 + 置信度）

---

## 开发方向与优先级

### 第一步：稳定编号逻辑（最高优先级）

这是设计文档反复强调的**业务稳定性关键点**。目前 `/detect` 返回的 `items` 列表顺序取决于 YOLO 内部输出顺序，没有确定性。

**需要实现的排序规则：**

- 先按中心点 `y` 坐标分层（用聚类或固定阈值）
- 同层内按 `x` 升序
- 层间按 `y` 升序
- 分配 A、B、C… 编号

**涉及的改动：**

| 文件 | 改动内容 |
|---|---|
| `app/services/detector.py` | 在 `_build_response` 中增加排序逻辑 |
| `app/schemas.py` | `DetectionItem` 增加 `label` 字段 |

**为什么优先级最高：**

- 同一张图多次调用时编号必须一致，否则 LLM 评分结果无法对应
- 这是后续标注图、裁剪图、LLM 评分的前提条件

---

### 第二步：标注图 + 裁剪图生成

设计文档要求将编号画到原图上，生成带标注的图片，同时裁剪出每个榴莲的局部图，交给后续 LLM 评分。

**标注图生成：**

- 用 PIL 或 OpenCV 在原图上绘制编号框 + 字母标签
- 返回 base64 或写入临时文件/对象存储后返回 URL

**裁剪图生成：**

- 根据检测框坐标裁剪原图
- 返回每个榴莲的裁剪图（base64 数组或 URL 列表）

**接口设计方案（二选一）：**

1. 新增 `POST /detect-and-annotate` 端点，一次性返回检测结果 + 标注图 + 裁剪图
2. 在现有 `/detect` 响应中增加 `annotated_image_base64` 和每个 item 的 `crop_image_base64` 字段

**为什么重要：**

LLM 评分同时需要带编号的整图和每个榴莲的裁剪图。没有这一步，就无法跑通 MVP 全链路。

---

### 第三步：补充自动化测试

当前项目没有任何测试。在进一步开发前建议补测试骨架。

**建议测试框架：** `pytest` + `pytest-asyncio`

**测试目录结构：**

```text
cv-service/
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_health.py
    ├── test_detect.py
    └── test_sorting.py
```

**需要覆盖的场景：**

| 测试类型 | 场景 |
|---|---|
| 单元测试 | `_build_response` 的排序/编号逻辑 |
| 单元测试 | schema 验证（BoundingBox、DetectionItem、DetectionResponse） |
| 集成测试 | `/health` 返回 `{"status": "ok"}` |
| 集成测试 | `/detect` 正常检测流程 |
| 边界用例 | 空文件上传 |
| 边界用例 | 非图片文件上传 |
| 边界用例 | 图片中无榴莲（返回 count=0） |
| 边界用例 | 图片中有大量榴莲（验证编号稳定性） |

---

### 第四步：输入校验增强

设计文档 `6.1 图片接入模块` 提到的校验规则目前完全没有实现。

**需要增加的校验：**

| 规则 | 说明 |
|---|---|
| 格式限制 | 只接受 `jpg/jpeg/png/webp` |
| 大小限制 | 不超过 10MB |
| 最小分辨率 | 宽度 ≥ 720px |
| 空文件检查 | 上传为空时返回友好错误 |

**实现建议：**

- 在 `detector.py` 的 `detect_upload` 方法中增加校验
- 或抽取为独立的 `validators.py` 模块

---

### 第五步：接口补充

| 接口 | 用途 |
|---|---|
| `POST /detect-and-annotate` | 检测 + 编号 + 标注图 + 裁剪图一次返回 |
| `GET /model-info` | 返回当前模型版本、类别信息、置信度阈值 |
| URL 输入支持 | `/detect` 除文件上传外，接受 `image_url` 参数 |

---

## 总结

| 阶段 | 任务 | 对 MVP 的影响 | 涉及文件 |
|---|---|---|---|
| **第一步** | 排序 + 编号逻辑 | 必须 — 解决编号漂移问题 | `detector.py`, `schemas.py` |
| **第二步** | 标注图 + 裁剪图生成 | 必须 — 打通 LLM 评分输入 | `detector.py`, `schemas.py`, 可能新增 `annotator.py` |
| **第三步** | 补自动化测试 | 重要 — 保障迭代质量 | 新增 `tests/` 目录 |
| **第四步** | 输入校验 | 重要 — 提升健壮性 | `detector.py` 或新增 `validators.py` |
| **第五步** | 接口整合与补充 | 可选 — 减少主后端调用次数 | `main.py`, `schemas.py` |

前两步完成后，cv-service 就能输出 LLM 评分所需的全部材料（编号列表 + 标注图 + 裁剪图），主后端（NestJS）可以开始对接 cv-service 并接入多模态模型，跑通 MVP 全链路。
