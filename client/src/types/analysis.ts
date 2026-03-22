export type AnalysisTaskStatus =
  | 'PENDING'
  | 'DETECTING'
  | 'SCORING'
  | 'DONE'
  | 'FAILED'

export interface CreateAnalysisTaskResponse {
  status: AnalysisTaskStatus
  taskId: string
  remainingCredits: number
}

export interface AnalysisTask {
  id: string
  createdAt: string
  sourceImagePath: string | null
  sourceImageUrl: string
  detectedCount: number
  detectedLabels: string[]
  status: AnalysisTaskStatus
  errorMessage: string | null
  overallSummary: string | null
  recommendedLabel: string | null
  rawResult: AnalysisTaskRawResult | null
  updatedAt: string
}

export interface AnalysisHistoryItem extends AnalysisTask {}

export interface AnalysisBoundingBox {
  x1: number
  x2: number
  y1: number
  y2: number
}

export interface AnalysisTaskDetectionItem {
  bbox: AnalysisBoundingBox
  confidence: number
  label: string
}

export interface AnalysisTaskRawResult {
  count: number
  items: AnalysisTaskDetectionItem[]
  message?: string | null
}

export interface AnalysisTaskItem extends AnalysisTaskDetectionItem {
  score: number | null
  summary: string | null
  reasons: string[] | null
  risks: string[] | null
  buyPriority: number | null
}

export interface AnalysisResult {
  sourceImageUrl: string
  recommendedLabel: string | null
  overallSummary: string | null
  items: AnalysisTaskItem[]
}
