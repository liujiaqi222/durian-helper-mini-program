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
  ReplaceAnalysisTaskItemsInput,
} from './durians.types';

@Injectable()
export class DrizzleDurianAnalysisRepository implements DurianAnalysisRepository {
  constructor(@Inject(DRIZZLE) private readonly db: DrizzleDb) {}

  async createTask(input: CreateAnalysisTaskInput): Promise<AnalysisTask> {
    const [task] = await this.db
      .insert(analysisTasks)
      .values({
        sourceImagePath: input.sourceImagePath ?? null,
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

  async replaceTaskItems(input: ReplaceAnalysisTaskItemsInput): Promise<void> {
    await this.db.delete(analysisTaskItems).where(eq(analysisTaskItems.taskId, input.taskId));

    if (input.items.length === 0) {
      return;
    }

    await this.db.insert(analysisTaskItems).values(
      input.items.map((item) => ({
        taskId: input.taskId,
        bbox: item.bbox,
        confidence: item.confidence,
        label: item.label,
        score: item.score,
        summary: item.summary,
        reasons: item.reasons,
        risks: item.risks,
        buyPriority: item.buyPriority,
      })),
    );
  }

  async updateTask(
    id: string,
    patch: Partial<AnalysisTask>,
  ): Promise<AnalysisTask | null> {
    const [task] = await this.db
      .update(analysisTasks)
      .set({
        detectedCount: patch.detectedCount,
        detectedLabels: patch.detectedLabels,
        errorMessage: patch.errorMessage,
        overallSummary: patch.overallSummary,
        rawResult: patch.rawResult,
        recommendedLabel: patch.recommendedLabel,
        sourceImagePath: patch.sourceImagePath,
        status: patch.status,
        updatedAt: new Date(),
      })
      .where(eq(analysisTasks.id, id))
      .returning();

    return task ? this.mapTask(task) : null;
  }

  private mapTask(task: typeof analysisTasks.$inferSelect): AnalysisTask {
    return {
      createdAt: task.createdAt,
      detectedCount: task.detectedCount,
      detectedLabels: Array.isArray(task.detectedLabels)
        ? (task.detectedLabels as string[])
        : [],
      errorMessage: task.errorMessage,
      id: task.id,
      overallSummary: task.overallSummary,
      rawResult: (task.rawResult as Record<string, unknown> | null) ?? null,
      recommendedLabel: task.recommendedLabel,
      sourceImagePath: task.sourceImagePath,
      sourceImageUrl: task.sourceImageUrl,
      status: task.status,
      updatedAt: task.updatedAt,
    };
  }

  private mapItem(
    item: typeof analysisTaskItems.$inferSelect,
  ): AnalysisTaskItem {
    return {
      bbox: item.bbox as AnalysisTaskItem['bbox'],
      confidence: item.confidence,
      buyPriority: item.buyPriority,
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
