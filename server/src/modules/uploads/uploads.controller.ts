import {
  BadRequestException,
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { UploadImageResponseDto } from './dto/upload-image-response.dto';
import { UploadedImageFile, UploadsService } from './uploads.service';

@Controller('uploads')
export class UploadsController {
  constructor(private readonly uploadsService: UploadsService) {}

  @Post('images')
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 10 * 1024 * 1024 } }))
  uploadImage(
    @UploadedFile() file: UploadedImageFile | undefined,
  ): Promise<UploadImageResponseDto> {
    if (!file) {
      throw new BadRequestException('file is required');
    }

    return this.uploadsService.storeUploadedFile(file);
  }
}
