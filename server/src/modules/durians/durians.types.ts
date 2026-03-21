export type AnalysisStatus =
  | 'PENDING'
  | 'DETECTING'
  | 'SCORING'
  | 'DONE'
  | 'FAILED';

export interface AnalysisBoundingBox {
  x1: number;
  x2: number;
  y1: number;
  y2: number;
}

export interface AnalysisTaskItem {
  id?: string;
  taskId?: string;
  bbox: AnalysisBoundingBox;
  confidence: number;
  label: string;
  score: number | null;
  summary: string | null;
  reasons: string[] | null;
  risks: string[] | null;
  buyPriority: number | null;
}

export interface AnalysisTask {
  id: string;
  sourceImagePath: string | null;
  sourceImageUrl: string;
  annotatedImageUrl: string | null;
  detectedCount: number;
  detectedLabels: string[];
  status: AnalysisStatus;
  errorMessage: string | null;
  overallSummary: string | null;
  recommendedLabel: string | null;
  rawResult: Record<string, unknown> | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface AnalysisTaskWithItems extends AnalysisTask {
  items: AnalysisTaskItem[];
}

export interface CreateAnalysisTaskInput {
  sourceImagePath?: string | null;
  sourceImageUrl: string;
}

export interface ReplaceAnalysisTaskItemsInput {
  taskId: string;
  items: AnalysisTaskItem[];
}

export interface DurianAnalysisRepository {
  createTask(input: CreateAnalysisTaskInput): Promise<AnalysisTask>;
  findTaskById(id: string): Promise<AnalysisTask | null>;
  findTaskResultById(id: string): Promise<AnalysisTaskWithItems | null>;
  replaceTaskItems(input: ReplaceAnalysisTaskItemsInput): Promise<void>;
  updateTask(
    id: string,
    patch: Partial<AnalysisTask>,
  ): Promise<AnalysisTask | null>;
}
