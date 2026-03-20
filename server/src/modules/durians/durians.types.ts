export type AnalysisStatus =
  | 'PENDING'
  | 'DETECTING'
  | 'SCORING'
  | 'DONE'
  | 'FAILED';

export interface AnalysisTaskItem {
  id?: string;
  taskId?: string;
  label: string;
  score: number | null;
  summary: string | null;
  reasons: string[] | null;
  risks: string[] | null;
  buyPriority: number | null;
  cropImageUrl: string | null;
}

export interface AnalysisTask {
  id: string;
  sourceImagePath: string | null;
  sourceImageUrl: string;
  annotatedImageUrl: string | null;
  status: AnalysisStatus;
  errorMessage: string | null;
  aiSummary: string | null;
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

export interface DurianAnalysisRepository {
  createTask(input: CreateAnalysisTaskInput): Promise<AnalysisTask>;
  findTaskById(id: string): Promise<AnalysisTask | null>;
  findTaskResultById(id: string): Promise<AnalysisTaskWithItems | null>;
  updateTask(
    id: string,
    patch: Partial<AnalysisTask>,
  ): Promise<AnalysisTask | null>;
}
