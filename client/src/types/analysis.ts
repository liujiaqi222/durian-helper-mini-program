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
  status: AnalysisTaskStatus
  errorMessage: string | null
  aiSummary: string | null
  recommendedLabel: string | null
}

export interface AnalysisTaskItem {
  label: string
  score: number | null
  summary: string | null
  reasons: string[] | null
  risks: string[] | null
  buyPriority: number | null
  cropImageUrl: string | null
}

export interface AnalysisResult {
  sourceImageUrl: string
  annotatedImageUrl: string | null
  recommendedLabel: string | null
  aiSummary: string | null
  items: AnalysisTaskItem[]
}
