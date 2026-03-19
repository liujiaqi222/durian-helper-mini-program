import { BadRequestException, ConflictException } from '@nestjs/common';
import { AiService } from '../ai/ai.service';
import { DuriansService } from './durians.service';
import type {
  AnalysisTask,
  CreateAnalysisTaskInput,
  DurianAnalysisRepository,
} from './durians.types';

class InMemoryDurianAnalysisRepository implements DurianAnalysisRepository {
  private readonly tasks = new Map<string, AnalysisTask>();

  createTask(input: CreateAnalysisTaskInput): Promise<AnalysisTask> {
    const task: AnalysisTask = {
      id: `task_${this.tasks.size + 1}`,
      sourceImageUrl: input.sourceImageUrl,
      annotatedImageUrl: null,
      status: 'PENDING',
      errorMessage: null,
      aiSummary: null,
      recommendedLabel: null,
      rawResult: null,
      createdAt: new Date('2026-03-19T00:00:00.000Z'),
      updatedAt: new Date('2026-03-19T00:00:00.000Z'),
    };

    this.tasks.set(task.id, task);
    return Promise.resolve(task);
  }

  findTaskById(id: string): Promise<AnalysisTask | null> {
    return Promise.resolve(this.tasks.get(id) ?? null);
  }

  findTaskResultById(id: string) {
    const task = this.tasks.get(id);
    if (!task) {
      return Promise.resolve(null);
    }
    return Promise.resolve({
      ...task,
      items: [],
    });
  }

  updateTask(
    id: string,
    patch: Partial<AnalysisTask>,
  ): Promise<AnalysisTask | null> {
    const existing = this.tasks.get(id);
    if (!existing) {
      return Promise.resolve(null);
    }

    const nextTask = {
      ...existing,
      ...patch,
      updatedAt: new Date('2026-03-19T00:01:00.000Z'),
    };

    this.tasks.set(id, nextTask);
    return Promise.resolve(nextTask);
  }
}

describe('DuriansService', () => {
  let repository: InMemoryDurianAnalysisRepository;
  let service: DuriansService;

  beforeEach(() => {
    repository = new InMemoryDurianAnalysisRepository();
    service = new DuriansService(repository, {
      summarizeDurianContext: jest.fn().mockResolvedValue('summary'),
    } as unknown as AiService);
  });

  it('creates a pending analysis task for an image url', async () => {
    const task = await service.createAnalysisTask({
      imageUrl: 'https://example.com/durian.png',
    });

    expect(task.id).toBe('task_1');
    expect(task.status).toBe('PENDING');
    expect(task.sourceImageUrl).toBe('https://example.com/durian.png');
  });

  it('rejects result lookup when the task is not completed', async () => {
    const task = await service.createAnalysisTask({
      imageUrl: 'https://example.com/durian.png',
    });

    await expect(service.getAnalysisResult(task.id)).rejects.toBeInstanceOf(
      ConflictException,
    );
  });

  it('retries only failed tasks', async () => {
    const task = await service.createAnalysisTask({
      imageUrl: 'https://example.com/durian.png',
    });

    await expect(service.retryAnalysisTask(task.id)).rejects.toBeInstanceOf(
      BadRequestException,
    );

    await repository.updateTask(task.id, {
      status: 'FAILED',
      errorMessage: 'cv-service timeout',
    });

    const retriedTask = await service.retryAnalysisTask(task.id);
    expect(retriedTask.status).toBe('PENDING');
    expect(retriedTask.errorMessage).toBeNull();
  });
});
