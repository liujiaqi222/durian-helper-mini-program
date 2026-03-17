# 榴莲 YOLO 训练与预测新手指南

这篇文档把仓库里原来两份训练文档合并成一条完整流程，目标是让第一次接触 YOLO 的同学也能回答清楚 4 个问题：

1. 我现在到底在训练什么
2. 从数据到 `best.pt` 的流程是什么
3. 每一步会产出什么文件
4. 训练完以后，怎样做预测并接回当前微服务

如果你是第一次做 CV，不要先追求“最高精度”。先把整条链路跑通，再根据误检、漏检去补数据和重训。

## 1. 先理解这件事在做什么

当前仓库训练的是“目标检测模型”，不是“分类模型”。

- 分类：判断一张图里有没有榴莲
- 检测：找出图里每一个榴莲的位置

这里必须做检测，因为后续服务需要每个榴莲的边界框，才能继续做：

1. 排序编号
2. 单果裁剪
3. 后续评分或解释

所以本阶段的目标非常明确：

- 输入：一张货架图或实拍图
- 输出：图片里每个榴莲的矩形框

## 2. 你要先记住的整条流程

先看全流程，再进入每一步细节：

```text
采集图片
  -> 标注榴莲框
  -> 整理成 YOLO 数据集目录
  -> 划分 train / val
  -> 执行训练
  -> 得到 runs/detect/train/weights/best.pt
  -> 用 best.pt 跑 predict 验证效果
  -> 复制到 models/durian-best.pt
  -> 由微服务加载并对外提供检测能力
```

如果只记“每步产物”，可以按这个表理解：

| 阶段 | 你在做什么 | 关键产物 |
| --- | --- | --- |
| 数据采集 | 收集真实榴莲图片 | 原始图片 |
| 标注 | 给每个榴莲画框 | YOLO 标签 `.txt` |
| 数据整理 | 放进固定目录结构 | `datasets/durian/` |
| 数据切分 | 分出训练集和验证集 | `images/train`、`images/val`、`labels/train`、`labels/val` |
| 训练 | 用 Ultralytics YOLO 学习 | `runs/detect/train/` |
| 选模型 | 使用最佳权重 | `runs/detect/train/weights/best.pt` |
| 预测验收 | 用新模型看新图/验证集 | `runs/detect/predict/` |
| 服务接入 | 替换线上/本地推理模型 | `models/durian-best.pt` |

## 3. 训练前需要准备什么

你至少需要这几样东西：

1. Python 3.11+ 环境
2. `cv-service` 的依赖
3. 一批榴莲图片
4. 标注工具

安装依赖的最小步骤：

```bash
cd cv-service
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. 数据应该长什么样

### 4.1 图片准备建议

第一版不需要上千张才能启动。对当前 MVP，建议这样理解：

- 最低可启动：100 到 200 张
- 更稳妥：300 到 800 张
- 继续优化：1000 张以上

比数量更重要的是覆盖真实场景：

- 多个榴莲同框
- 遮挡
- 远近不同
- 光线偏暗、偏黄、偏绿
- 背景复杂
- 角度变化

优先级也很明确：

1. 真实业务图片
2. 接近业务场景的图片
3. 公开图片

原因很简单：训练时看到的场景越接近上线场景，模型越不容易“离线看起来可以，线上就失真”。

### 4.2 标注规则

当前任务只有一个类别：

```text
durian
```

不要一开始拆成好果、坏果、开口、闭口。第一阶段只做“找到榴莲在哪里”。

标注时遵守这几条规则：

1. 框住完整榴莲主体，边缘尽量贴合，但不要为了像素级精确耗费太多时间。
2. 能明确判断是独立榴莲的遮挡目标，也要标。
3. 只露出一点、连人都难判断是不是独立榴莲的，不要硬标。
4. 同一批数据的标注标准要一致，尤其是框的松紧和遮挡是否标注。

### 4.3 YOLO 数据目录

当前仓库的数据集目录是：

```text
cv-service/datasets/durian/
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   └── val
└── data.yaml
```

它的含义是：

- `images/train`：训练图片
- `labels/train`：训练标签
- `images/val`：验证图片
- `labels/val`：验证标签
- `data.yaml`：告诉 YOLO 去哪里读数据、类别叫什么

图片和标签必须同名，例如：

```text
images/train/001.jpg
labels/train/001.txt
```

### 4.4 标签文件长什么样

YOLO 标签文件是纯文本，每一行表示一个框，例如：

```text
0 0.512 0.487 0.301 0.442
0 0.242 0.530 0.220 0.390
```

你通常不需要手写。标注工具会导出。

只要知道：

- 第一列 `0` 是类别 ID
- 因为这里只有 `durian` 一个类别，所以它一直是 `0`
- 后面 4 个数是归一化后的框坐标和宽高

### 4.5 `data.yaml`

当前仓库已经使用下面这个配置：

```yaml
path: datasets/durian
train: images/train
val: images/val

names:
  0: durian
```



## 5. 这个仓库如何切分 train / val

当前仓库已经提供了切分脚本：

```bash
cd cv-service
source .venv/bin/activate
python3 scripts/split_yolo_dataset.py
```

这个脚本会做三件事：

1. 先把旧的 `val` 文件收回 `train`
2. 再按固定随机种子重新切分
3. 保证图片和标签一起移动，不会拆散

这样做的原因不是“炫技”，而是为了避免两个常见问题：

- 你每次手工挪文件，容易把图片和标签挪错
- 你补完新图再切分时，如果直接在旧结果上继续切，会越来越乱

脚本执行后的产物还是同一个目录，只是其中的文件被重新分配：

```text
cv-service/datasets/durian/images/train/
cv-service/datasets/durian/images/val/
cv-service/datasets/durian/labels/train/
cv-service/datasets/durian/labels/val/
```

如果脚本能正常执行，你会在终端看到类似：

```text
split complete: total=103 train=82 val=21 seed=42
```

这表示：

- 当前总共有多少组有效图片/标签
- 其中多少进训练集
- 多少进验证集

## 6. 第一次训练前要检查什么

正式训练前只检查 4 件事：

1. `images/train` 和 `labels/train` 是否一一对应
2. `images/val` 和 `labels/val` 是否一一对应
3. `datasets/durian/data.yaml` 是否仍然指向当前目录结构
4. 类别是否仍然只有一个 `durian`

如果你重新导出了标签，优先重新执行一次切分脚本，再开始训练。

## 7. 如何开始训练

### 7.1 最小训练命令

如果你的目标是先把流程跑通，直接使用这条：

```bash
cd cv-service
source .venv/bin/activate
yolo detect train \
  data="datasets/durian/data.yaml" \
  model="yolov8n.pt" \
  project="runs" \
  name="train" \
  epochs=50 \
  imgsz=640 \
  batch=4 \
  device=cpu
```

这里各参数的核心含义：

- `data`：数据集配置
- `model`：训练起点，这里用轻量的 `yolov8n.pt`
- `project` + `name`：训练结果写到 `runs/detect/train/`
- `epochs=50`：先跑一版，验证流程和数据质量
- `imgsz=640`：输入尺寸
- `batch=4`：CPU 环境更稳妥的起步值
- `device=cpu`：强制 CPU，避免新人一开始卡在 GPU 环境

### 7.2 如果你可以使用 GPU

确认机器有可用 GPU 后，可以改成：

```bash
cd cv-service
source .venv/bin/activate
yolo detect train \
  data="datasets/durian/data.yaml" \
  model="yolov8n.pt" \
  project="runs" \
  name="train" \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  device=0
```

不要把“更大的模型”当成第一优先级。对新人来说，先验证数据和标注质量，收益通常比盲目加大模型更高。

## 8. 训练过程中会产生什么

训练输出目录默认是：

```text
cv-service/runs/detect/train/
```

其中最常见的文件包括：

```text
cv-service/runs/detect/train/
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
├── labels.jpg
├── train_batch*.jpg
├── val_batch*_pred.jpg
└── weights
    ├── best.pt
    └── last.pt
```

你可以这样理解这些产物：

- `weights/best.pt`
  - 验证集表现最好的权重
  - 这是你后面最关心的文件
- `weights/last.pt`
  - 最后一个 epoch 的权重
  - 不一定比 `best.pt` 更好
- `results.csv`
  - 每轮训练的指标记录
- `results.png`
  - 指标趋势图，方便快速看收敛情况
- `confusion_matrix.png`
  - 类别混淆情况图
  - 当前只有一个类别，更多是辅助排查异常
- `train_batch*.jpg`
  - 训练样本可视化
  - 用来确认框和图片读入是否正常
- `val_batch*_pred.jpg`
  - 验证集预测可视化
  - 用来直观看漏检和误检

对当前项目，训练完成后你最需要记住的是这一个文件：

```text
cv-service/runs/detect/train/weights/best.pt
```

## 9. 训练完以后怎样做预测

训练完成以后，不要先埋头看一堆指标。第一轮验收先看模型“画框是否靠谱”。

### 9.1 用验证集做批量预测

```bash
cd cv-service
source .venv/bin/activate
yolo detect predict \
  model="runs/detect/train/weights/best.pt" \
  source="datasets/durian/images/val" \
  conf=0.35 \
  project="runs" \
  name="predict"
```

这条命令的含义：

- `model`：使用你刚训练出来的最佳权重
- `source`：输入图片来源，这里用验证集
- `conf=0.35`：置信度阈值，低于它的框会被过滤
- `project` + `name`：结果输出到 `runs/detect/predict/`

预测产物通常会出现在：

```text
cv-service/runs/detect/predict/
```

目录里会保存已经画好框的图片，例如：

```text
cv-service/runs/detect/predict/033.jpg
cv-service/runs/detect/predict/079.jpg
```

这些图就是“可视化验收结果”。你要看的不是抽象指标，而是：

1. 大多数榴莲有没有被框出来
2. 框的位置是否基本合理
3. 是否把大量背景或其他物体误识别成榴莲

### 9.2 用单张新图做预测

如果你手上有一张没参与训练的新图，可以这样试：

```bash
cd cv-service
source .venv/bin/activate
yolo detect predict \
  model="runs/detect/train/weights/best.pt" \
  source="path/to/your-test-image.jpg" \
  conf=0.35 \
  project="runs" \
  name="predict-single"
```

这样更接近真实上线场景，也更容易发现模型是否只记住了训练集背景。

## 10. 如何判断下一步该补数据还是调参数

如果你看到下面这些问题，优先补数据和修标注，不要急着折腾参数：

1. 遮挡场景漏检很多
2. 密集摆放时只识别出一部分
3. 换一批光线条件后效果明显变差
4. 某些拍摄角度几乎不识别

因为这类问题通常不是“学习率调错了”，而是训练数据没有覆盖真实场景。

## 11. 怎样把模型接回当前微服务

当前微服务默认从下面这个路径加载模型：

```text
cv-service/models/durian-best.pt
```

所以训练验收通过后，把 `best.pt` 复制过去即可：

```bash
cd cv-service
cp runs/detect/train/weights/best.pt models/durian-best.pt
```

这一步的产物是：

```text
cv-service/models/durian-best.pt
```

然后启动服务：

```bash
cd cv-service
source .venv/bin/activate
uvicorn app.main:app --reload --port 8010
```

服务启动后，会在启动阶段加载这个模型文件。

## 12. 服务里的“predict”是什么样

命令行的 `yolo detect predict` 适合做训练后的可视化验收。

而当前仓库里的 Python 微服务，负责的是“接收一张图片，返回检测框 JSON”。两者区别如下：

| 方式 | 适用场景 | 输入 | 输出 |
| --- | --- | --- | --- |
| `yolo detect predict` | 训练后人工验收 | 本地图片/目录 | 画好框的图片 |
| 微服务 `/detect` | 给其他系统调用 | 上传图片 | JSON 检测框 |

当前服务的职责是：

1. 读取 `models/durian-best.pt`
2. 对上传图片做 YOLO 推理
3. 返回每个榴莲的坐标和置信度

也就是说：

- CLI `predict` 看的是“模型视觉效果”
- 微服务 `/detect` 提供的是“程序可消费的数据”

## 13. 新人最稳的实际执行顺序

如果你是第一次接手这个模块，按下面顺序做最稳：

1. 安装依赖并进入 `cv-service/.venv`
2. 准备图片和标注
3. 确认 `datasets/durian/` 目录结构正确
4. 运行 `python3 scripts/split_yolo_dataset.py`
5. 运行第一版训练命令
6. 拿 `runs/detect/train/weights/best.pt` 跑 `predict`
7. 人工看 `runs/detect/predict/` 里的结果图
8. 满意后复制到 `models/durian-best.pt`
9. 启动微服务验证 `/detect`

## 14. 最容易踩的坑

### 14.1 只看训练效果，不看验证集或新图

训练集好看，不代表真实场景也好看。至少要看 `val`，更好的是再看几张新图。

### 14.2 手工移动图片和标签

这很容易把一对文件拆开。当前仓库已经有 `scripts/split_yolo_dataset.py`，优先使用脚本。

### 14.3 标注标准前后不一致

如果第一批图框得很紧，第二批图框得很松，模型会学得很混乱。补数据时一定沿用同一套标注标准。

### 14.4 一开始就急着调很多参数

第一版的主要目标是确认：

- 数据能读
- 标注没问题
- 模型能收敛
- 预测结果方向正确

在这之前，大量调参通常只是放大噪音。

## 15. 一句话总结

对当前仓库来说，最重要的主线只有这一条：

```text
准备并标注图片 -> 切分数据集 -> 训练 -> 得到 best.pt -> 跑 predict 验收 -> 复制到 models/durian-best.pt -> 微服务加载使用
```

如果你只能记住一个“关键产物”，那就是：

```text
cv-service/runs/detect/train/weights/best.pt
```

它是训练阶段的最终模型产物，也是接回服务的输入。
