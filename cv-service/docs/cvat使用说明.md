# CVAT 本地标注环境

这个目录放的是当前项目的 `CVAT` 本地使用说明，目标是：

1. 用 Docker 在本机启动 `CVAT`
2. 把现有数据目录挂载进 `CVAT` 的共享目录
3. 让你直接在网页里给榴莲图片画框
4. 最后导出为 YOLO 可训练的数据

下面的命令示例统一约定：

```bash
CV_DIR=cv-service
```

## 目录约定

当前项目里的榴莲数据目录是：

```text
cv-service/datasets/durian/
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
```

本地 `CVAT` 环境会把下面这个目录挂载到容器内的 `/home/django/share`：

```text
cv-service/datasets
```

这样在 `CVAT` 里创建任务时，可以直接从共享目录里选择：

```text
durian/images/train
```

## 一次性准备

先执行：

```bash
CV_DIR=cv-service
bash "$CV_DIR/scripts/cvat_local.sh" prepare
```

这一步会做两件事：

1. 如果本地还没有 `CVAT` 源码，就克隆官方仓库到：

```text
cv-service/.tools/cvat
```

2. 生成本项目专用的 `docker-compose.override.yml`，把你的数据目录挂进去

## 启动与停止

启动：

```bash
CV_DIR=cv-service
bash "$CV_DIR/scripts/cvat_local.sh" start
```

停止：

```bash
CV_DIR=cv-service
bash "$CV_DIR/scripts/cvat_local.sh" stop
```

查看状态：

```bash
CV_DIR=cv-service
bash "$CV_DIR/scripts/cvat_local.sh" status
```

默认访问地址：

```text
http://localhost:8080
```

## 标注建议

第一阶段只保留一个类别：

```text
durian
```

画框时遵守下面几条即可：

1. 框住整个榴莲主体，边缘尽量贴合
2. 被遮挡但仍明显是独立榴莲的，也标
3. 完全看不清、你自己也不确定的，不要硬标
4. 同一批图保持同一标准，不要时紧时松

## 在 CVAT 里怎么建任务

1. 打开 `http://localhost:8080`
2. 创建一个项目，例如 `durian-train`
3. 标签只建一个：`durian`
4. 新建任务时选择从 `Share` 导入文件
5. 进入共享目录 `durian/images/train`
6. 选中训练图片，开始标框

## 导出到 YOLO

标完之后，从任务或项目里导出为：

```text
YOLO 1.1
```

导出后需要把结果整理成：

```text
cv-service/datasets/durian/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── data.yaml
```

说明：

1. `CVAT` 导出的 YOLO 压缩包里通常会带图片和标签
2. 你的训练目录里已经有图片了，所以重点是把 `.txt` 标签放回 `labels/train` 或 `labels/val`
3. 标签文件名必须和图片文件名同名，例如 `001.jpg` 对应 `001.txt`

## 为什么用共享目录挂载

这里不用每次在网页里手工上传图片，而是直接把本地数据目录挂载给 `CVAT`。这样做有几个好处：

1. 图片多时更稳定
2. 重建任务时不用反复上传
3. 后续 `train / val` 目录切分后还能直接复用
4. 目录结构始终和项目训练目录保持一致，后面不容易整理乱

## 参考

官方安装说明：

- [CVAT Installation Guide](https://docs.cvat.ai/latest/docs/administration/basics/installation/)

官方共享目录说明：

- [CVAT Share Path](https://docs.cvat.ai/latest/docs/administration/basics/installation/#share-path)
