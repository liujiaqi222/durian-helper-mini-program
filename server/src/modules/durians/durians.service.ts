import {
  BadRequestException,
  ConflictException,
  Inject,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { LoggerService } from '../../core/logger/logger.service';
import { UsersService } from '../users/users.service';
import { AiService } from '../ai/ai.service';
import { UploadsService } from '../uploads/uploads.service';
import { DURIAN_ANALYSIS_REPOSITORY } from './durians.constants';
import { CvService } from './cv.service';
import type {
  AnalysisTask,
  AnalysisTaskItem,
  AnalysisTaskWithItems,
  DurianAnalysisRepository,
} from './durians.types';

@Injectable()
export class DuriansService {
  private readonly activeTaskIds = new Set<string>();

  constructor(
    @Inject(DURIAN_ANALYSIS_REPOSITORY)
    private readonly repository: DurianAnalysisRepository,
    private readonly aiService: AiService,
    private readonly cvService: CvService,
    private readonly uploadsService: UploadsService,
    private readonly usersService: UsersService,
    private readonly logger: LoggerService,
  ) {}

  async createAnalysisTask(input: {
    userId: number;
    imagePath?: string;
    imageUrl: string;
  }): Promise<AnalysisTask & { remainingCredits: number }> {
    const userProfile = await this.usersService.consumeAnalyzeCredit(input.userId);
    const task = await this.repository.createTask({
      userId: input.userId,
      sourceImagePath: input.imagePath,
      sourceImageUrl: input.imageUrl,
    });

    this.logger.log(
      `Created durian analysis task ${JSON.stringify({
        imagePath: input.imagePath ?? null,
        imageUrl: input.imageUrl,
        taskId: task.id,
      })}`,
      'DuriansService',
    );

    this.scheduleTaskProcessing(task.id);
    return {
      ...task,
      remainingCredits: userProfile.remainingCredits,
    };
  }

  async getAnalysisTask(taskId: string): Promise<AnalysisTask> {
    const task = await this.repository.findTaskById(taskId);
    if (!task) {
      throw new NotFoundException(`Task ${taskId} not found`);
    }
    return this.hydrateTaskUrls(task);
  }

  async getHistoryTasks(userId: number): Promise<AnalysisTask[]> {
    const tasks = await this.repository.findRecentTasksByUserId(userId, 20);
    return tasks.map((task) => this.hydrateTaskUrls(task));
  }

  async getAnalysisResult(taskId: string): Promise<AnalysisTaskWithItems> {
    const task = await this.repository.findTaskResultById(taskId);
    if (!task) {
      throw new NotFoundException(`Task ${taskId} not found`);
    }
    if (task.status !== 'DONE') {
      throw new ConflictException('Analysis result is not ready yet');
    }
    return this.hydrateTaskUrls(task);
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

    await this.repository.replaceTaskItems({
      taskId: task.id,
      items: [],
    });

    const nextTask = await this.repository.updateTask(task.id, {
      detectedCount: 0,
      detectedLabels: [],
      errorMessage: null,
      overallSummary: null,
      rawResult: null,
      recommendedLabel: null,
      status: 'PENDING',
    });

    if (!nextTask) {
      throw new NotFoundException(`Task ${taskId} not found`);
    }

    this.scheduleTaskProcessing(task.id);
    return this.hydrateTaskUrls(nextTask);
  }

  private scheduleTaskProcessing(taskId: string): void {
    if (this.activeTaskIds.has(taskId)) {
      return;
    }

    this.activeTaskIds.add(taskId);
    setTimeout(() => {
      void this.processTask(taskId).finally(() => {
        this.activeTaskIds.delete(taskId);
      });
    }, 0);
  }

  private async processTask(taskId: string): Promise<void> {
    const task = await this.getAnalysisTask(taskId);
    const imagePath = task.sourceImagePath ?? undefined;
    const imageUrl = this.uploadsService.buildPublicUrl(task.sourceImageUrl)!;

    await this.repository.updateTask(task.id, {
      errorMessage: null,
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
        imagePath,
        imageUrl,
        taskId: task.id,
      });
      const detectedLabels = detectionResult.items.map((item) => item.label);

      await this.repository.updateTask(task.id, {
        detectedCount: detectionResult.count,
        detectedLabels,
        rawResult: this.buildPersistedRawResult(detectionResult),
        status: 'SCORING',
      });
      this.logger.log(
        `CV detection completed ${JSON.stringify({
          count: detectionResult.count,
          itemLabels: detectedLabels,
          taskId: task.id,
        })}`,
        'DuriansService',
      );

      const scored = await this.aiService.scoreDurians(
        detectionResult.items.map((item) => ({
          bbox: item.bbox,
          confidence: item.confidence,
          cropImageBase64: item.cropImageBase64,
          imagePath,
          imageUrl,
          label: item.label,
        })),
      );

      const taskItems: AnalysisTaskItem[] = detectionResult.items.map((item) => {
        const score = scored.items.find((candidate) => candidate.label === item.label);
        if (!score) {
          throw new Error(`Missing AI score for detected label ${item.label}`);
        }

        return {
          bbox: item.bbox,
          confidence: item.confidence,
          label: item.label,
          score: score.score,
          summary: score.summary,
          reasons: score.reasons,
          risks: score.risks,
          buyPriority: score.buyPriority,
        };
      });

      await this.repository.replaceTaskItems({
        taskId: task.id,
        items: taskItems,
      });

      await this.repository.updateTask(task.id, {
        overallSummary: scored.overallSummary,
        recommendedLabel: scored.recommendedLabel,
        status: 'DONE',
      });
      this.logger.log(
        `Durian analysis task completed ${JSON.stringify({
          recommendedLabel: scored.recommendedLabel,
          taskId: task.id,
        })}`,
        'DuriansService',
      );
    } catch (error) {
      this.logger.error(
        `Durian analysis task failed ${JSON.stringify({
          imageUrl,
          taskId: task.id,
        })}`,
        error instanceof Error ? error.stack : undefined,
        'DuriansService',
      );
      await this.repository.replaceTaskItems({
        taskId: task.id,
        items: [],
      });
      await this.repository.updateTask(task.id, {
        errorMessage:
          error instanceof Error ? error.message : 'durian analysis failed',
        overallSummary: null,
        recommendedLabel: null,
        status: 'FAILED',
      });
    }
  }

  private buildPersistedRawResult(
    detectionResult: Awaited<ReturnType<CvService['detectAndAnnotate']>>,
  ): Record<string, unknown> {
    return {
      count: detectionResult.count,
      items: detectionResult.items.map((item) => ({
        bbox: item.bbox,
        class_name: item.class_name,
        confidence: item.confidence,
        label: item.label,
      })),
      message: detectionResult.message ?? null,
    };
  }

  private hydrateTaskUrls<T extends AnalysisTask | AnalysisTaskWithItems>(
    task: T,
  ): T {
    return {
      ...task,
      rawResult: task.rawResult,
      sourceImageUrl: task.sourceImageUrl ?? '',
    };
  }
}
