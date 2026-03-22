import { BadRequestException, ConflictException } from '@nestjs/common';
import { AiService } from '../ai/ai.service';
import { LoggerService } from '../../core/logger/logger.service';
import { CvService } from './cv.service';
import { DuriansService } from './durians.service';
import { UploadsService } from '../uploads/uploads.service';
import type {
  AnalysisTask,
  AnalysisTaskItem,
  CreateAnalysisTaskInput,
  DurianAnalysisRepository,
  ReplaceAnalysisTaskItemsInput,
} from './durians.types';
import { UsersService } from '../users/users.service';

class InMemoryDurianAnalysisRepository implements DurianAnalysisRepository {
  private readonly tasks = new Map<string, AnalysisTask & { items: AnalysisTaskItem[] }>();

  createTask(input: CreateAnalysisTaskInput): Promise<AnalysisTask> {
    const task: AnalysisTask & { items: AnalysisTaskItem[] } = {
      id: `task_${this.tasks.size + 1}`,
      userId: input.userId,
      sourceImagePath: input.sourceImagePath ?? null,
      sourceImageUrl: input.sourceImageUrl,
      detectedCount: 0,
      detectedLabels: [],
      status: 'PENDING',
      errorMessage: null,
      overallSummary: null,
      recommendedLabel: null,
      rawResult: null,
      items: [],
      createdAt: new Date('2026-03-19T00:00:00.000Z'),
      updatedAt: new Date('2026-03-19T00:00:00.000Z'),
    };

    this.tasks.set(task.id, task);
    return Promise.resolve({ ...task, items: undefined } as unknown as AnalysisTask);
  }

  findTaskById(id: string): Promise<AnalysisTask | null> {
    const task = this.tasks.get(id);
    if (!task) {
      return Promise.resolve(null);
    }

    return Promise.resolve({ ...task, items: undefined } as unknown as AnalysisTask);
  }

  findTaskResultById(id: string) {
    const task = this.tasks.get(id);
    if (!task) {
      return Promise.resolve(null);
    }
    return Promise.resolve({
      ...task,
      items: task.items.map((item) => ({ ...item })),
    });
  }

  findRecentTasksByUserId(userId: number, limit: number): Promise<AnalysisTask[]> {
    return Promise.resolve(
      [...this.tasks.values()]
        .filter((task) => task.userId === userId)
        .sort((left, right) => right.createdAt.getTime() - left.createdAt.getTime())
        .slice(0, limit)
        .map((task) => ({ ...task, items: undefined } as unknown as AnalysisTask)),
    );
  }

  replaceTaskItems(input: ReplaceAnalysisTaskItemsInput): Promise<void> {
    const task = this.tasks.get(input.taskId);
    if (!task) {
      return Promise.resolve();
    }

    this.tasks.set(input.taskId, {
      ...task,
      items: input.items.map((item) => ({ ...item, taskId: input.taskId })),
    });
    return Promise.resolve();
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
    return Promise.resolve({ ...nextTask, items: undefined } as unknown as AnalysisTask);
  }
}

describe('DuriansService', () => {
  let repository: InMemoryDurianAnalysisRepository;
  let service: DuriansService;
  let cvService: { detectAndAnnotate: jest.Mock };
  let logger: { log: jest.Mock; warn: jest.Mock; error: jest.Mock };
  let aiService: { scoreDurians: jest.Mock };
  let uploadsService: { buildPublicUrl: jest.Mock };
  let usersService: { consumeAnalyzeCredit: jest.Mock };

  beforeEach(() => {
    jest.useFakeTimers();
    repository = new InMemoryDurianAnalysisRepository();
    cvService = {
      detectAndAnnotate: jest.fn().mockResolvedValue({
        count: 1,
        items: [
          {
            bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
            class_name: 'durian',
            confidence: 0.92,
            cropImageBase64: 'ZmFrZS1jcm9w',
            label: 'A',
          },
        ],
      }),
    };
    logger = {
      log: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
    };
    uploadsService = {
      buildPublicUrl: jest.fn((value: string | null | undefined) => {
        if (!value) {
          return null;
        }
        if (value.startsWith('http')) {
          return value;
        }
        return `http://localhost:3000${value}`;
      }),
    };
    aiService = {
      scoreDurians: jest.fn().mockResolvedValue({
        overallSummary: 'A 综合表现最好。',
        recommendedLabel: 'A',
        items: [
          {
            label: 'A',
            score: 92,
            summary: '编号 A 更适合买。',
            reasons: ['外形完整', '刺分布均匀'],
            risks: ['仅凭图片无法判断内部状态'],
            buyPriority: 1,
          },
        ],
      }),
    };
    usersService = {
      consumeAnalyzeCredit: jest.fn().mockResolvedValue({
        remainingCredits: 9,
      }),
    };
    service = new DuriansService(
      repository,
      aiService as unknown as AiService,
      cvService as unknown as CvService,
      uploadsService as unknown as UploadsService,
      usersService as unknown as UsersService,
      logger as unknown as LoggerService,
    );
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  async function flushBackgroundTask(): Promise<void> {
    await jest.runOnlyPendingTimersAsync();
  }

  it('creates a task immediately and completes it asynchronously', async () => {
    const task = await service.createAnalysisTask({
      imageUrl: '/uploads/durian.png',
    });

    expect(task.id).toBe('task_1');
    expect(task.status).toBe('PENDING');

    await flushBackgroundTask();

    const storedTask = await service.getAnalysisTask(task.id);
    expect(storedTask.status).toBe('DONE');
    expect(storedTask.sourceImageUrl).toBe('/uploads/durian.png');
    expect(storedTask.detectedCount).toBe(1);
    expect(storedTask.detectedLabels).toEqual(['A']);
    expect(storedTask.overallSummary).toBe('A 综合表现最好。');
    expect(storedTask.recommendedLabel).toBe('A');
    expect(storedTask.rawResult).toEqual({
      count: 1,
      items: [
        {
          bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
          class_name: 'durian',
          confidence: 0.92,
          label: 'A',
        },
      ],
      message: null,
    });
    expect(cvService.detectAndAnnotate).toHaveBeenCalledWith({
      imagePath: undefined,
      imageUrl: 'http://localhost:3000/uploads/durian.png',
      taskId: 'task_1',
    });
    expect(aiService.scoreDurians).toHaveBeenCalledWith([
      expect.objectContaining({
        cropImageBase64: 'ZmFrZS1jcm9w',
        imageUrl: 'http://localhost:3000/uploads/durian.png',
        label: 'A',
      }),
    ]);
  });

  it('persists uploaded image path and saves final scored items', async () => {
    const task = await service.createAnalysisTask({
      imagePath: '/tmp/uploads/task_1.jpg',
      imageUrl: '/uploads/task_1.jpg',
    });

    await flushBackgroundTask();

    const result = await service.getAnalysisResult(task.id);
    expect(result.items).toEqual([
      {
        bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
        confidence: 0.92,
        label: 'A',
        score: 92,
        summary: '编号 A 更适合买。',
        reasons: ['外形完整', '刺分布均匀'],
        risks: ['仅凭图片无法判断内部状态'],
        buyPriority: 1,
        taskId: 'task_1',
      },
    ]);
    expect(cvService.detectAndAnnotate).toHaveBeenCalledWith({
      imagePath: '/tmp/uploads/task_1.jpg',
      imageUrl: 'http://localhost:3000/uploads/task_1.jpg',
      taskId: 'task_1',
    });
  });

  it('rejects result lookup when the task is not completed', async () => {
    const task = await service.createAnalysisTask({
      imageUrl: '/uploads/durian.png',
    });

    await expect(service.getAnalysisResult(task.id)).rejects.toBeInstanceOf(
      ConflictException,
    );
  });

  it('retries only failed tasks', async () => {
    const task = await service.createAnalysisTask({
      imageUrl: '/uploads/durian.png',
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
      imageUrl: '/uploads/durian.png',
    });

    await flushBackgroundTask();

    const storedTask = await service.getAnalysisTask(task.id);
    expect(storedTask.status).toBe('FAILED');
    expect(storedTask.errorMessage).toBe('cv down');
    expect(logger.error).toHaveBeenCalledWith(
      expect.stringContaining('Durian analysis task failed'),
      expect.any(String),
      'DuriansService',
    );
  });

  it('marks the task as failed when ai scoring returns invalid output', async () => {
    aiService.scoreDurians.mockRejectedValueOnce(new Error('invalid ai json'));

    const task = await service.createAnalysisTask({
      imageUrl: '/uploads/durian.png',
    });

    await flushBackgroundTask();

    const storedTask = await service.getAnalysisTask(task.id);
    expect(storedTask.status).toBe('FAILED');
    expect(storedTask.errorMessage).toBe('invalid ai json');
  });

  it('returns the latest 20 tasks for the given user only', async () => {
    for (let index = 0; index < 22; index += 1) {
      const task = await service.createAnalysisTask({
        userId: 101,
        imageUrl: `/uploads/history-${index}.png`,
      });

      await repository.updateTask(task.id, {
        createdAt: new Date(`2026-03-19T00:${String(index).padStart(2, '0')}:00.000Z`),
      });
    }

    await service.createAnalysisTask({
      userId: 202,
      imageUrl: '/uploads/other-user.png',
    });

    const history = await service.getHistoryTasks(101);

    expect(history).toHaveLength(20);
    expect(history.every((task) => task.userId === 101)).toBe(true);
    expect(history[0]?.sourceImageUrl).toBe('/uploads/history-21.png');
    expect(history[19]?.sourceImageUrl).toBe('/uploads/history-2.png');
  });
});
