import { randomUUID } from 'crypto';
import { Injectable } from '@nestjs/common';
import type {
  AnalysisTask,
  AnalysisTaskWithItems,
  CreateAnalysisTaskInput,
  DurianAnalysisRepository,
} from './durians.types';

@Injectable()
export class InMemoryDurianAnalysisRepository implements DurianAnalysisRepository {
  private readonly tasks = new Map<string, AnalysisTaskWithItems>();

  createTask(input: CreateAnalysisTaskInput): Promise<AnalysisTask> {
    const now = new Date();
    const task: AnalysisTaskWithItems = {
      id: randomUUID(),
      sourceImageUrl: input.sourceImageUrl,
      annotatedImageUrl: null,
      status: 'PENDING',
      errorMessage: null,
      aiSummary: null,
      recommendedLabel: null,
      rawResult: null,
      items: [],
      createdAt: now,
      updatedAt: now,
    };

    this.tasks.set(task.id, task);
    return Promise.resolve(this.cloneTask(task));
  }

  findTaskById(id: string): Promise<AnalysisTask | null> {
    const task = this.tasks.get(id);
    return Promise.resolve(task ? this.cloneTask(task) : null);
  }

  findTaskResultById(id: string): Promise<AnalysisTaskWithItems | null> {
    const task = this.tasks.get(id);
    if (!task) {
      return Promise.resolve(null);
    }

    return Promise.resolve({
      ...this.cloneTask(task),
      items: task.items.map((item) => ({ ...item })),
    });
  }

  updateTask(
    id: string,
    patch: Partial<AnalysisTask>,
  ): Promise<AnalysisTask | null> {
    const task = this.tasks.get(id);
    if (!task) {
      return Promise.resolve(null);
    }

    const nextTask: AnalysisTaskWithItems = {
      ...task,
      ...patch,
      items: task.items,
      updatedAt: new Date(),
    };

    this.tasks.set(id, nextTask);
    return Promise.resolve(this.cloneTask(nextTask));
  }

  private cloneTask(task: AnalysisTask): AnalysisTask {
    return {
      ...task,
      createdAt: new Date(task.createdAt),
      updatedAt: new Date(task.updatedAt),
    };
  }
}
