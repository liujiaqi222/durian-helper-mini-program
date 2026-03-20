# 榴莲识别小程序 MVP 方案

## 目标

在 `client/` 下实现一个可联调的小程序 MVP，打通这条最短业务链路：

1. 用户选择或拍摄一张榴莲货架图片
2. 小程序把图片上传到主后端
3. 小程序发起榴莲分析任务
4. 小程序轮询任务状态
5. 小程序展示推荐结果和全部评分列表

本期只做 MVP，不做分享卡、历史记录、登录态和复杂视觉包装。

## 页面拆分

采用双页结构，保证代码可读、职责清晰：

### 1. 首页 `pages/index/index`

职责：

1. 选择图片
2. 预览本地图片
3. 上传图片到后端
4. 发起分析任务
5. 跳转到结果页

首页不承担轮询和结果渲染，避免单页状态过重。

### 2. 结果页 `pages/result/index`

职责：

1. 根据 `taskId` 轮询任务状态
2. 展示分析进度
3. 任务完成后拉取最终结果
4. 展示推荐榴莲和全部评分列表
5. 在失败态提供重试和重新选择图片入口

## 状态管理

使用 `zustand` 做一个轻量全局 store，只保存当前分析上下文，不做复杂状态架构。

建议状态字段：

- `localImagePath`
- `uploadedImageUrl`
- `uploadedImagePath`
- `taskId`
- `taskStatus`
- `result`
- `errorMessage`

状态流转：

- `idle`
- `uploading`
- `pending`
- `detecting`
- `scoring`
- `done`
- `failed`

## 目录设计

```text
client/src/
├── pages/
│   ├── index/
│   └── result/
├── services/
│   └── api.ts
├── store/
│   └── analysis.ts
├── types/
│   └── analysis.ts
└── utils/
    └── analysis.ts
```

职责划分：

- `pages/` 只处理页面交互和渲染
- `services/api.ts` 统一封装后端请求
- `store/analysis.ts` 管理分析流程状态
- `types/analysis.ts` 放接口与页面共用类型
- `utils/analysis.ts` 放状态文案、轮询判断等纯函数

## 接口联调约定

根据当前后端实现，前端按以下顺序调用：

### 1. 上传图片

接口：

- `POST /uploads/images`

用途：

- 上传用户选中的本地图片

预期返回：

- `fileName`
- `fileUrl`
- `localPath`

### 2. 创建分析任务

接口：

- `POST /durians/analyze`

请求体：

```json
{
  "imageUrl": "https://example.com/uploads/xxx.jpg",
  "imagePath": "/abs/path/to/local/file.jpg"
}
```

预期返回：

```json
{
  "taskId": "task_xxx",
  "status": "SCORING"
}
```

说明：

- 当前后端会在接口内部先触发 CV 检测流程，再返回任务状态
- 前端仍然统一进入结果页，并以任务详情接口为准更新状态

### 3. 查询任务状态

接口：

- `GET /durians/tasks/{taskId}`

状态枚举：

- `PENDING`
- `DETECTING`
- `SCORING`
- `DONE`
- `FAILED`

### 4. 获取分析结果

接口：

- `GET /durians/tasks/{taskId}/result`

调用时机：

- 只有任务状态为 `DONE` 时才调用

结果页需展示这些核心字段：

- `sourceImageUrl`
- `annotatedImageUrl`
- `recommendedLabel`
- `aiSummary`
- `items`

每个 `item` 重点展示：

- `label`
- `score`
- `summary`
- `reasons`
- `risks`
- `buyPriority`
- `cropImageUrl`

### 5. 失败重试

接口：

- `POST /durians/tasks/{taskId}/retry`

MVP 用法：

- 结果页失败态可调用该接口后继续轮询
- 若重试失败，也允许用户返回首页重新上传

## 页面交互设计

### 首页

展示内容：

1. 产品标题和简短说明
2. 图片选择区域
3. 已选图片预览
4. “开始分析”按钮

交互规则：

1. 未选图时禁用“开始分析”
2. 点击开始分析后，显示上传中状态，避免重复点击
3. 上传成功并创建任务后跳转到结果页
4. 上传或建任务失败时展示错误提示

### 结果页

展示内容：

1. 当前任务状态
2. 原图预览
3. 标注图预览
4. 推荐榴莲卡片
5. 全部评分列表
6. 失败态按钮

交互规则：

1. `PENDING / DETECTING / SCORING` 显示进度文案
2. `DONE` 后自动拉取结果并渲染
3. `FAILED` 显示错误信息和重试按钮
4. 支持返回首页重新选图

## 可读性约束

实现时遵守以下规则：

1. 页面组件不直接拼接复杂请求逻辑
2. store 只做流程状态协调，不承担具体网络实现
3. 类型单独定义，避免页面里出现大量匿名对象
4. 工具函数保持纯函数，避免隐藏副作用
5. 优先少量小文件，不写超长单文件页面

## MVP 之外暂不实现

以下能力本期明确不做：

1. 分享海报生成
2. 历史分析记录
3. 用户登录与鉴权
4. 多图上传
5. 复杂筛选和图表可视化
6. 生产级埋点与监控

## 开发与联调

安装依赖：

```bash
cd client
npm install
```

开发小程序：

```bash
npm run dev:weapp
```

构建小程序：

```bash
npm run build:weapp
```

联调前需要确保：

1. NestJS 主后端已经启动
2. `cv-service` 已可用
3. 前端请求基地址已配置为当前后端地址

## 后续迭代建议

MVP 跑通后，再按优先级继续做：

1. 分享图与结果页包装
2. 历史记录页
3. 重试与超时策略优化
4. 更细的错误分类
5. 登录态与云存储适配
