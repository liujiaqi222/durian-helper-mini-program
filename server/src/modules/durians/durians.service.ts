import {
  BadRequestException,
  ConflictException,
  Inject,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { LoggerService } from '../../core/logger/logger.service';
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
    private readonly logger: LoggerService,
  ) {}

  async createAnalysisTask(input: {
    imagePath?: string;
    imageUrl: string;
  }): Promise<AnalysisTask> {
    const task = await this.repository.createTask({
      sourceImagePath: input.imagePath,
      sourceImageUrl: input.imageUrl,
    });
    this.logger.log(
      `Starting durian analysis task ${JSON.stringify({
        imagePath: input.imagePath ?? null,
        imageUrl: input.imageUrl,
        taskId: task.id,
      })}`,
      'DuriansService',
    );

    await this.repository.updateTask(task.id, {
      sourceImagePath: input.imagePath ?? null,
      status: 'DETECTING',
    });
    this.logger.log(
      `Durian analysis task moved to DETECTING ${JSON.stringify({
        taskId: task.id,
      })}`,
      'DuriansService',
    );

    try {
      const detectionResult = await this.cvService.detectAndAnnotate({
        imagePath: input.imagePath,
        imageUrl: input.imageUrl,
        taskId: task.id,
      });
      this.logger.log(
        `CV detection completed ${JSON.stringify({
          annotatedImageUrl: detectionResult.annotatedImageUrl,
          count: detectionResult.count,
          itemLabels: detectionResult.items.map((item) => item.label),
          taskId: task.id,
        })}`,
        'DuriansService',
      );
      const aiSummary = await this.aiService
        .summarizeDurianContext(input.imageUrl)
        .catch((error) => {
          this.logger.error(
            `AI summary generation failed ${JSON.stringify({
              imageUrl: input.imageUrl,
              taskId: task.id,
            })}`,
            error instanceof Error ? error.stack : undefined,
            'DuriansService',
          );
          return null;
        });

      if (aiSummary) {
        this.logger.log(
          `AI summary received ${JSON.stringify({
            summary: aiSummary,
            taskId: task.id,
          })}`,
          'DuriansService',
        );
      }

      return (
        (await this.repository.updateTask(task.id, {
          aiSummary,
          annotatedImageUrl: detectionResult.annotatedImageUrl,
          rawResult: detectionResult as unknown as Record<string, unknown>,
          status: 'SCORING',
        })) ?? task
      );
    } catch (error) {
      this.logger.error(
        `Durian analysis task failed ${JSON.stringify({
          imageUrl: input.imageUrl,
          taskId: task.id,
        })}`,
        error instanceof Error ? error.stack : undefined,
        'DuriansService',
      );
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
    this.logger.log(
      `Retrying durian analysis task ${JSON.stringify({ taskId })}`,
      'DuriansService',
    );

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
