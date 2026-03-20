import {
  BadRequestException,
  ConflictException,
  Inject,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { AiService } from '../ai/ai.service';
import { DURIAN_ANALYSIS_REPOSITORY } from './durians.constants';
import { CvService } from './cv.service';
import type {
  AnalysisTask,
  AnalysisTaskWithItems,
  DurianAnalysisRepository,
} from './durians.types';

@Injectable()
export class DuriansService {
  constructor(
    @Inject(DURIAN_ANALYSIS_REPOSITORY)
    private readonly repository: DurianAnalysisRepository,
    private readonly aiService: AiService,
    private readonly cvService: CvService,
  ) {}

  async createAnalysisTask(input: {
    imagePath?: string;
    imageUrl: string;
  }): Promise<AnalysisTask> {
    const task = await this.repository.createTask({
      sourceImagePath: input.imagePath,
      sourceImageUrl: input.imageUrl,
    });

    await this.repository.updateTask(task.id, {
      sourceImagePath: input.imagePath ?? null,
      status: 'DETECTING',
    });

    try {
      const detectionResult = await this.cvService.detectAndAnnotate({
        imagePath: input.imagePath,
        imageUrl: input.imageUrl,
        taskId: task.id,
      });
      const aiSummary = await this.aiService
        .summarizeDurianContext(input.imageUrl)
        .catch(() => null);

      return (
        (await this.repository.updateTask(task.id, {
          aiSummary,
          annotatedImageUrl: detectionResult.annotatedImageUrl,
          rawResult: detectionResult as unknown as Record<string, unknown>,
          status: 'SCORING',
        })) ?? task
      );
    } catch (error) {
      return (
        (await this.repository.updateTask(task.id, {
          errorMessage:
            error instanceof Error ? error.message : 'cv-service request failed',
          status: 'FAILED',
        })) ?? task
      );
    }
  }

  async getAnalysisTask(taskId: string): Promise<AnalysisTask> {
    const task = await this.repository.findTaskById(taskId);
    if (!task) {
      throw new NotFoundException(`Task ${taskId} not found`);
    }
    return task;
  }

  async getAnalysisResult(taskId: string): Promise<AnalysisTaskWithItems> {
    const task = await this.repository.findTaskResultById(taskId);
    if (!task) {
      throw new NotFoundException(`Task ${taskId} not found`);
    }
    if (task.status !== 'DONE') {
      throw new ConflictException('Analysis result is not ready yet');
    }
    return task;
  }

  async retryAnalysisTask(taskId: string): Promise<AnalysisTask> {
    const task = await this.getAnalysisTask(taskId);
    if (task.status !== 'FAILED') {
      throw new BadRequestException('Only failed tasks can be retried');
    }

    const nextTask = await this.repository.updateTask(task.id, {
      aiSummary: null,
      errorMessage: null,
      rawResult: null,
      recommendedLabel: null,
      status: 'PENDING',
    });

    if (!nextTask) {
      throw new NotFoundException(`Task ${taskId} not found`);
    }

    return nextTask;
  }
}
