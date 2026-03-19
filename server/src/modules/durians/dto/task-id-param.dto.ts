import { IsUUID } from 'class-validator';

export class TaskIdParamDto {
  @IsUUID()
  taskId!: string;
}
