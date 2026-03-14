# 榴莲 YOLO 训练落地指南

这份文档只覆盖当前仓库里的这套数据和训练路径，目标是把你已经标好的榴莲框尽快训练成第一版可用模型。

下面的命令示例统一约定：

```bash
CV_DIR=python-cv-service
```

## 1. 当前数据状态

当前数据集目录：

```text
python-cv-service/datasets/durian/
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   └── val
└── data.yaml
```

数据切分由脚本生成：

```bash
CV_DIR=python-cv-service
python3 "$CV_DIR/scripts/split_yolo_dataset.py"
```

默认规则：

1. 先把 `val` 里的文件收回到 `train`
2. 再按固定随机种子重新切分
3. 保证图片和同名标签一起移动

这样做的原因是：后面你补新图或重做切分时，结果可重复，不会越切越乱。

## 2. 第一次训练前要确认什么

训练前只检查下面 4 项：

1. `images/train` 和 `labels/train` 文件名一一对应
2. `images/val` 和 `labels/val` 文件名一一对应
3. `data.yaml` 指向当前数据目录
4. 类别只有一个：`durian`

如果你重新导出了标签，优先重新运行切分脚本，再开始训练。

## 3. 启动训练

先进入 Python 虚拟环境：

```bash
CV_DIR=python-cv-service
cd "$CV_DIR"
source .venv/bin/activate
```

### 3.1 CPU 最稳妥起步命令

如果你只是想先把流程跑通，用这条：

```bash
yolo detect train \
  data=datasets/durian/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=4 \
  device=cpu
```

### 3.2 如果你机器能用 GPU

如果后面你确认有可用 GPU，可以把 `device=cpu` 去掉，或改成对应设备编号：

```bash
yolo detect train \
  data=datasets/durian/data.yaml \
  model=yolov8n.pt \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  device=0
```

这里先用 `yolov8n.pt`，是因为它轻、快，适合第一版验证数据和标注质量。

## 4. 训练结果在哪里

Ultralytics 默认会输出到：

```text
runs/detect/train/
```

你最关心的是：

```text
runs/detect/train/weights/best.pt
```

这就是后面接回微服务的模型文件。

## 5. 训练后怎样快速验收

先不要一开始就盯着很多指标。第一版只看三件事：

1. 大多数榴莲能不能被框出来
2. 框的位置是否大致合理
3. 有没有明显把背景或别的水果误识别成榴莲

你可以直接跑一次预测检查：

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=datasets/durian/images/val \
  conf=0.35
```

结果通常会出现在：

```text
runs/detect/predict/
```

## 6. 什么时候该继续补数据

如果出现下面这些情况，优先补数据而不是急着调参数：

1. 遮挡场景漏检很多
2. 密集摆放时只框出一部分
3. 光线变化后效果明显变差
4. 某些角度或距离几乎不识别

这是因为你现在的数据量还比较小，先天更容易过拟合到有限场景。

## 7. 接回当前微服务

当你确认 `best.pt` 效果还可以时，复制到：

```text
python-cv-service/models/durian-best.pt
```

当前微服务默认就是按这个路径读取模型。

## 8. 推荐的实际执行顺序

按这个顺序最稳：

1. 确认数据切分已经完成
2. 跑一次 `yolov8n` 小模型训练
3. 用 `val` 或少量新图做预测检查
4. 判断问题主要来自“数据”还是“参数”
5. 需要时再补图、重标、重训

## 9. 常见坑

### 9.1 只看训练集效果，不看验证集

训练图上效果很好，不代表新图也好。一定要看 `val` 或额外测试图。

### 9.2 切分后手工挪文件

手工挪文件很容易把图片和标签拆开。后续如果要重做切分，直接重新运行脚本。

### 9.3 标注标准前后不一致

如果后面补图，框的松紧、遮挡是否标注、边缘贴合程度，都要尽量沿用同一标准。
