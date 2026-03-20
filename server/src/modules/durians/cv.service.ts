import { readFile } from 'fs/promises';
import { basename, extname } from 'path';
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { LoggerService } from '../../core/logger/logger.service';
import { UploadsService } from '../uploads/uploads.service';

interface CvDetectionItem {
  bbox: {
    x1: number;
    x2: number;
    y1: number;
    y2: number;
  };
  class_name: string;
  confidence: number;
  crop_image_base64?: string | null;
  label: string;
}

interface CvDetectionResponse {
  annotated_image_base64?: string | null;
  count: number;
  items: CvDetectionItem[];
  message?: string | null;
}

export interface DetectAndAnnotateInput {
  imagePath?: string;
  imageUrl: string;
  taskId: string;
}

export interface DetectAndAnnotateOutput {
  annotatedImageUrl: string | null;
  count: number;
  items: Array<{
    bbox: CvDetectionItem['bbox'];
    class_name: string;
    confidence: number;
    cropImageUrl: string | null;
    label: string;
  }>;
  message?: string | null;
}

@Injectable()
export class CvService {
  constructor(
    private readonly configService: ConfigService,
    private readonly uploadsService: UploadsService,
    private readonly logger: LoggerService,
  ) {}

  async detectAndAnnotate(
    input: DetectAndAnnotateInput,
  ): Promise<DetectAndAnnotateOutput> {
    const response = await this.callDetectAndAnnotate(input);

    const annotatedImageUrl = response.annotated_image_base64
      ? (
          await this.uploadsService.storeBase64Image(
            response.annotated_image_base64,
            `${input.taskId}-annotated`,
          )
        ).fileUrl
      : null;

    const items = await Promise.all(
      response.items.map(async (item) => {
        const cropImageUrl = item.crop_image_base64
          ? (
              await this.uploadsService.storeBase64Image(
                item.crop_image_base64,
                `${input.taskId}-${item.label}`,
              )
            ).fileUrl
          : null;

        return {
          bbox: item.bbox,
          class_name: item.class_name,
          confidence: item.confidence,
          cropImageUrl,
          label: item.label,
        };
      }),
    );

    return {
      annotatedImageUrl,
      count: response.count,
      items,
      message: response.message ?? null,
    };
  }

  private async callDetectAndAnnotate(
    input: DetectAndAnnotateInput,
  ): Promise<CvDetectionResponse> {
    const endpoint = `${this.configService.get<string>('cvService.baseUrl')}/detect-and-annotate`;
    const formData = new FormData();

    if (input.imagePath) {
      const buffer = await readFile(input.imagePath);
      const extension = extname(input.imagePath).toLowerCase() || '.jpg';
      const mimeType = this.getMimeType(extension);
      formData.append(
        'file',
        new Blob([buffer], { type: mimeType }),
        basename(input.imagePath),
      );
    } else {
      formData.append('image_url', input.imageUrl);
    }

    this.logger.log(
      `Calling cv-service detect-and-annotate ${JSON.stringify({
        endpoint,
        hasImagePath: Boolean(input.imagePath),
        imagePath: input.imagePath ?? null,
        imageUrl: input.imageUrl,
        taskId: input.taskId,
      })}`,
      'CvService',
    );

    const response = await fetch(endpoint, {
      body: formData,
      method: 'POST',
    });

    if (!response.ok) {
      this.logger.error(
        `cv-service detect-and-annotate failed ${JSON.stringify({
          endpoint,
          status: response.status,
          taskId: input.taskId,
        })}`,
        undefined,
        'CvService',
      );
      throw new Error(`cv-service returned ${response.status}`);
    }

    const payload = (await response.json()) as CvDetectionResponse;
    this.logger.log(
      `cv-service detect-and-annotate completed ${JSON.stringify({
        annotatedImageReturned: Boolean(payload.annotated_image_base64),
        count: payload.count,
        itemLabels: payload.items.map((item) => item.label),
        message: payload.message ?? null,
        taskId: input.taskId,
      })}`,
      'CvService',
    );
    return payload;
  }

  private getMimeType(extension: string): string {
    switch (extension) {
      case '.png':
        return 'image/png';
      case '.webp':
        return 'image/webp';
      default:
        return 'image/jpeg';
    }
  }
}
