export type AnalysisTaskStatus =
  | 'PENDING'
  | 'DETECTING'
  | 'SCORING'
  | 'DONE'
  | 'FAILED'

export interface CreateAnalysisTaskResponse {
  status: AnalysisTaskStatus
  taskId: string
}

export interface AnalysisTask {
  id: string
  sourceImagePath: string | null
  sourceImageUrl: string
  annotatedImageUrl: string | null
  detectedCount: number
  detectedLabels: string[]
  status: AnalysisTaskStatus
  errorMessage: string | null
  overallSummary: string | null
  recommendedLabel: string | null
}

export interface AnalysisTaskItem {
  bbox: {
    x1: number
    x2: number
    y1: number
    y2: number
  }
  confidence: number
  label: string
  score: number | null
  summary: string | null
  reasons: string[] | null
  risks: string[] | null
  buyPriority: number | null
}

export interface AnalysisResult {
  sourceImageUrl: string
  annotatedImageUrl: string | null
  recommendedLabel: string | null
  overallSummary: string | null
  items: AnalysisTaskItem[]
}
