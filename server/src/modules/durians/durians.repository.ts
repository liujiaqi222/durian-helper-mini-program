import { Inject, Injectable } from '@nestjs/common';
import { eq } from 'drizzle-orm';
import { DRIZZLE, type DrizzleDb } from '../../database/drizzle/drizzle.module';
import {
  analysisTaskItems,
  analysisTasks,
} from '../../database/drizzle/schema';
import type {
  AnalysisTask,
  AnalysisTaskItem,
  AnalysisTaskWithItems,
  CreateAnalysisTaskInput,
  DurianAnalysisRepository,
} from './durians.types';

@Injectable()
export class DrizzleDurianAnalysisRepository implements DurianAnalysisRepository {
  constructor(@Inject(DRIZZLE) private readonly db: DrizzleDb) {}

  async createTask(input: CreateAnalysisTaskInput): Promise<AnalysisTask> {
    const [task] = await this.db
      .insert(analysisTasks)
      .values({
        sourceImageUrl: input.sourceImageUrl,
      })
      .returning();

    return this.mapTask(task);
  }

  async findTaskById(id: string): Promise<AnalysisTask | null> {
    const [task] = await this.db
      .select()
      .from(analysisTasks)
      .where(eq(analysisTasks.id, id))
      .limit(1);

    return task ? this.mapTask(task) : null;
  }

  async findTaskResultById(id: string): Promise<AnalysisTaskWithItems | null> {
    const [task] = await this.db
      .select()
      .from(analysisTasks)
      .where(eq(analysisTasks.id, id))
      .limit(1);

    if (!task) {
      return null;
    }

    const items = await this.db
      .select()
      .from(analysisTaskItems)
      .where(eq(analysisTaskItems.taskId, id));

    return {
      ...this.mapTask(task),
      items: items.map((item) => this.mapItem(item)),
    };
  }

  async updateTask(
    id: string,
    patch: Partial<AnalysisTask>,
  ): Promise<AnalysisTask | null> {
    const [task] = await this.db
      .update(analysisTasks)
      .set({
        aiSummary: patch.aiSummary,
        annotatedImageUrl: patch.annotatedImageUrl,
        errorMessage: patch.errorMessage,
        rawResult: patch.rawResult,
        recommendedLabel: patch.recommendedLabel,
        status: patch.status,
        updatedAt: new Date(),
      })
      .where(eq(analysisTasks.id, id))
      .returning();

    return task ? this.mapTask(task) : null;
  }

  private mapTask(task: typeof analysisTasks.$inferSelect): AnalysisTask {
    return {
      aiSummary: task.aiSummary,
      annotatedImageUrl: task.annotatedImageUrl,
      createdAt: task.createdAt,
      errorMessage: task.errorMessage,
      id: task.id,
      rawResult: (task.rawResult as Record<string, unknown> | null) ?? null,
      recommendedLabel: task.recommendedLabel,
      sourceImageUrl: task.sourceImageUrl,
      status: task.status,
      updatedAt: task.updatedAt,
    };
  }

  private mapItem(
    item: typeof analysisTaskItems.$inferSelect,
  ): AnalysisTaskItem {
    return {
      buyPriority: item.buyPriority,
      cropImageUrl: item.cropImageUrl,
      id: item.id,
      label: item.label,
      reasons: (item.reasons as string[] | null) ?? null,
      risks: (item.risks as string[] | null) ?? null,
      score: item.score,
      summary: item.summary,
      taskId: item.taskId,
    };
  }
}
