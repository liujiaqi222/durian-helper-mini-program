import {
  BadRequestException,
  Controller,
  Get,
  Param,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { LoggerService } from '../../core/logger/logger.service';
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
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 10 * 1024 * 1024 } }))
  async createAnalysisTask(@UploadedFile() file: UploadedImageFile | undefined) {
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
        fileUrl: storedImage.fileUrl,
        localPath: storedImage.localPath,
      })}`,
      'DuriansController',
    );
    const task = await this.duriansService.createAnalysisTask({
      imagePath: storedImage.localPath,
      imageUrl: storedImage.fileUrl,
    });
    this.logger.log(
      `Durian analyze request completed ${JSON.stringify({
        status: task.status,
        taskId: task.id,
      })}`,
      'DuriansController',
    );
    return {
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
