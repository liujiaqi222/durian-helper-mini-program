import { IsUrl } from 'class-validator';

export class CreateAnalysisTaskDto {
  @IsUrl({
    require_protocol: true,
  })
  imageUrl!: string;
}
