import { BadRequestException, ConflictException } from '@nestjs/common';
import { AiService } from '../ai/ai.service';
import { CvService } from './cv.service';
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
      sourceImagePath: input.sourceImagePath ?? null,
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
  let cvService: { detectAndAnnotate: jest.Mock };

  beforeEach(() => {
    repository = new InMemoryDurianAnalysisRepository();
    cvService = {
      detectAndAnnotate: jest.fn().mockResolvedValue({
        annotatedImageUrl: 'http://localhost:3000/uploads/annotated-task_1.jpg',
        count: 1,
        items: [
          {
            bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
            class_name: 'durian',
            confidence: 0.92,
            cropImageUrl: 'http://localhost:3000/uploads/crops/task_1-A.jpg',
            label: 'A',
          },
        ],
      }),
    };
    service = new DuriansService(repository, {
      summarizeDurianContext: jest.fn().mockResolvedValue('summary'),
    } as unknown as AiService, cvService as unknown as CvService);
  });

  it('creates an analysis task and stores cv output for an image url', async () => {
    const task = await service.createAnalysisTask({
      imageUrl: 'https://example.com/durian.png',
    });

    expect(task.id).toBe('task_1');
    expect(task.status).toBe('SCORING');
    expect(task.sourceImageUrl).toBe('https://example.com/durian.png');
    expect(task.annotatedImageUrl).toBe(
      'http://localhost:3000/uploads/annotated-task_1.jpg',
    );
    expect(task.rawResult).toEqual({
      annotatedImageUrl: 'http://localhost:3000/uploads/annotated-task_1.jpg',
      count: 1,
      items: [
        {
          bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
          class_name: 'durian',
          confidence: 0.92,
          cropImageUrl: 'http://localhost:3000/uploads/crops/task_1-A.jpg',
          label: 'A',
        },
      ],
    });
    expect(cvService.detectAndAnnotate).toHaveBeenCalledWith({
      imagePath: undefined,
      imageUrl: 'https://example.com/durian.png',
      taskId: 'task_1',
    });
  });

  it('persists the uploaded image path and prefers it for cv detection', async () => {
    const task = await service.createAnalysisTask({
      imagePath: '/tmp/uploads/task_1.jpg',
      imageUrl: 'http://localhost:3000/uploads/task_1.jpg',
    });

    expect(task.sourceImagePath).toBe('/tmp/uploads/task_1.jpg');
    expect(cvService.detectAndAnnotate).toHaveBeenCalledWith({
      imagePath: '/tmp/uploads/task_1.jpg',
      imageUrl: 'http://localhost:3000/uploads/task_1.jpg',
      taskId: 'task_1',
    });
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

  it('marks the task as failed when cv detection throws', async () => {
    cvService.detectAndAnnotate.mockRejectedValueOnce(new Error('cv down'));

    const task = await service.createAnalysisTask({
      imageUrl: 'https://example.com/durian.png',
    });

    expect(task.status).toBe('FAILED');
    expect(task.errorMessage).toBe('cv down');
  });
});
