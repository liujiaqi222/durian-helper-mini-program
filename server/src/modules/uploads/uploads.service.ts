import { mkdir, writeFile } from 'fs/promises';
import { extname, join, resolve } from 'path';
import { BadRequestException, Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { UploadImageResponseDto } from './dto/upload-image-response.dto';

const UPLOAD_DIR_NAME = 'uploads';
const IMAGE_MIME_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/jpg',
]);
const IMAGE_EXTENSION_BY_MIME: Record<string, string> = {
  'image/jpg': '.jpg',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
};

export interface UploadedImageFile {
  buffer: Buffer;
  mimetype: string;
  originalname: string;
}

@Injectable()
export class UploadsService {
  constructor(private readonly configService: ConfigService) {}

  getUploadsDir(): string {
    return resolve(process.cwd(), UPLOAD_DIR_NAME);
  }

  async storeUploadedFile(file: UploadedImageFile): Promise<UploadImageResponseDto> {
    this.ensureSupportedImage(file.mimetype);

    const extension =
      extname(file.originalname).toLowerCase() ||
      IMAGE_EXTENSION_BY_MIME[file.mimetype] ||
      '.jpg';
    const fileName = this.buildFileName('upload', extension);
    return this.writeBuffer(file.buffer, fileName);
  }

  async storeBase64Image(
    base64Value: string,
    prefix: string,
  ): Promise<UploadImageResponseDto> {
    const { buffer, extension } = this.decodeBase64Image(base64Value);
    const fileName = this.buildFileName(prefix, extension);
    return this.writeBuffer(buffer, fileName);
  }

  private async writeBuffer(
    buffer: Buffer,
    fileName: string,
  ): Promise<UploadImageResponseDto> {
    const uploadsDir = this.getUploadsDir();
    await mkdir(uploadsDir, { recursive: true });

    const localPath = join(uploadsDir, fileName);
    await writeFile(localPath, buffer);

    return {
      fileName,
      fileUrl: `${this.configService.get<string>('publicBaseUrl')}/uploads/${fileName}`,
      localPath,
    };
  }

  private ensureSupportedImage(mimeType: string): void {
    if (!IMAGE_MIME_TYPES.has(mimeType)) {
      throw new BadRequestException('Only jpg, png, and webp images are supported.');
    }
  }

  private buildFileName(prefix: string, extension: string): string {
    const safePrefix = prefix.replace(/[^a-zA-Z0-9_-]/g, '-');
    return `${safePrefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}${extension}`;
  }

  private decodeBase64Image(base64Value: string): {
    buffer: Buffer;
    extension: string;
  } {
    const dataUrlMatch = base64Value.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
    if (dataUrlMatch) {
      const mimeType = dataUrlMatch[1];
      this.ensureSupportedImage(mimeType);
      return {
        buffer: Buffer.from(dataUrlMatch[2], 'base64'),
        extension: IMAGE_EXTENSION_BY_MIME[mimeType] || '.jpg',
      };
    }

    return {
      buffer: Buffer.from(base64Value, 'base64'),
      extension: '.jpg',
    };
  }
}
