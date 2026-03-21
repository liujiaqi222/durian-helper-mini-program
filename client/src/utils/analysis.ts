import type { AnalysisResult, AnalysisTaskItem, AnalysisTaskStatus } from '../types/analysis'

const statusDescriptionMap: Record<AnalysisTaskStatus, string> = {
  PENDING: '任务已创建，准备开始分析',
  DETECTING: '正在识别榴莲位置和编号',
  SCORING: 'AI 正在生成评分和购买建议',
  DONE: '分析完成',
  FAILED: '分析失败，请重试',
}

export function getStatusDescription(status: AnalysisTaskStatus): string {
  return statusDescriptionMap[status]
}

export function isTerminalTaskStatus(status: AnalysisTaskStatus): boolean {
  return status === 'DONE' || status === 'FAILED'
}

function compareCandidatePriority(a: AnalysisTaskItem, b: AnalysisTaskItem): number {
  const priorityA = a.buyPriority ?? Number.MAX_SAFE_INTEGER
  const priorityB = b.buyPriority ?? Number.MAX_SAFE_INTEGER

  if (priorityA !== priorityB) {
    return priorityA - priorityB
  }

  const scoreA = a.score ?? Number.MIN_SAFE_INTEGER
  const scoreB = b.score ?? Number.MIN_SAFE_INTEGER

  return scoreB - scoreA
}

export function findRecommendedItem(result: AnalysisResult): AnalysisTaskItem | null {
  if (result.recommendedLabel) {
    const matchedItem = result.items.find((item) => item.label === result.recommendedLabel)
    if (matchedItem) {
      return matchedItem
    }
  }

  if (result.items.length === 0) {
    return null
  }

  return [...result.items].sort(compareCandidatePriority)[0] ?? null
}

export function sortItemsForDisplay(items: AnalysisTaskItem[]): AnalysisTaskItem[] {
  return [...items].sort(compareCandidatePriority)
}

interface ResultPreviewInput {
  annotatedImageUrl: string | null
  sourceImageUrl: string | null
  localImagePath: string | null
}

interface ResultPreview {
  title: '编号标注图' | '原图'
  imageUrl: string
}

export function resolveResultPreview(input: ResultPreviewInput): ResultPreview | null {
  if (input.annotatedImageUrl) {
    return {
      title: '编号标注图',
      imageUrl: input.annotatedImageUrl,
    }
  }

  if (input.sourceImageUrl) {
    return {
      title: '原图',
      imageUrl: input.sourceImageUrl,
    }
  }

  if (input.localImagePath) {
    return {
      title: '原图',
      imageUrl: input.localImagePath,
    }
  }

  return null
}
