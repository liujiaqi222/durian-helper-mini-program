import { Body, Controller, Get, Param, Post } from '@nestjs/common';
import { CreateAnalysisTaskDto } from './dto/create-analysis-task.dto';
import { TaskIdParamDto } from './dto/task-id-param.dto';
import { DuriansService } from './durians.service';

@Controller('durians')
export class DuriansController {
  constructor(private readonly duriansService: DuriansService) {}

  @Post('analyze')
  async createAnalysisTask(@Body() dto: CreateAnalysisTaskDto) {
    const task = await this.duriansService.createAnalysisTask(dto);
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
