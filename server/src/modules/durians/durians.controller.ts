import {
  BadRequestException,
  Controller,
  Get,
  Param,
  Post,
  UploadedFile,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { LoggerService } from '../../core/logger/logger.service';
import { AuthGuard } from '../auth/auth.guard';
import { CurrentUser } from '../auth/current-user.decorator';
import type { AuthenticatedUser } from '../auth/auth.types';
import { TaskIdParamDto } from './dto/task-id-param.dto';
import { DuriansService } from './durians.service';
import { UploadedImageFile, UploadsService } from '../uploads/uploads.service';

@Controller('durians')
export class DuriansController {
  constructor(
    private readonly duriansService: DuriansService,
    private readonly uploadsService: UploadsService,
    private readonly logger: LoggerService,
  ) {}

  @Post('analyze')
  @UseGuards(AuthGuard)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 10 * 1024 * 1024 } }))
  async createAnalysisTask(
    @CurrentUser() user: AuthenticatedUser,
    @UploadedFile() file: UploadedImageFile | undefined,
  ) {
    if (!file) {
      throw new BadRequestException('file is required');
    }

    this.logger.log(
      `Received durian analyze request ${JSON.stringify({
        mimeType: file.mimetype,
        originalName: file.originalname,
        size: file.buffer.length,
      })}`,
      'DuriansController',
    );

    const storedImage = await this.uploadsService.storeUploadedFile(file);
    this.logger.log(
      `Uploaded image stored ${JSON.stringify({
        filePath: storedImage.filePath,
        fileUrl: storedImage.fileUrl,
        localPath: storedImage.localPath,
      })}`,
      'DuriansController',
    );
    const task = await this.duriansService.createAnalysisTask({
      userId: user.userId,
      imagePath: storedImage.localPath,
      imageUrl: storedImage.filePath,
    });
    this.logger.log(
      `Durian analyze request completed ${JSON.stringify({
        status: task.status,
        taskId: task.id,
      })}`,
      'DuriansController',
    );
    return {
      remainingCredits: task.remainingCredits,
      status: task.status,
      taskId: task.id,
    };
  }

  @Get('tasks/:taskId')
  getAnalysisTask(@Param() params: TaskIdParamDto) {
    return this.duriansService.getAnalysisTask(params.taskId);
  }

  @Get('tasks/:taskId/result')
  getAnalysisResult(@Param() params: TaskIdParamDto) {
    return this.duriansService.getAnalysisResult(params.taskId);
  }

  @Post('tasks/:taskId/retry')
  retryAnalysisTask(@Param() params: TaskIdParamDto) {
    return this.duriansService.retryAnalysisTask(params.taskId);
  }
}
