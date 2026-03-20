import { IsOptional, IsString, IsUrl } from 'class-validator';

export class CreateAnalysisTaskDto {
  @IsUrl({
    require_protocol: true,
  })
  imageUrl!: string;

  @IsOptional()
  @IsString()
  imagePath?: string;
}
