import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { readFile } from 'fs/promises';
import { extname } from 'path';
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';
import { LoggerService } from '../../core/logger/logger.service';

const IMAGE_MEDIA_TYPE_BY_EXTENSION: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
};

@Injectable()
export class AiService {
  constructor(
    private readonly configService: ConfigService,
    private readonly logger: LoggerService,
  ) {}

  async summarizeDurianContext(input: {
    imagePath?: string | null;
    imageUrl: string;
  }): Promise<string> {
    const apiKey = this.configService.get<string>('ai.apiKey');
    if (!apiKey) {
      this.logger.warn(
        `Skipping durian AI summary because ai.apiKey is missing ${JSON.stringify(
          {
            imagePath: input.imagePath ?? null,
            imageUrl: input.imageUrl,
          },
        )}`,
        'AiService',
      );
      return `AI not configured for ${input.imageUrl}`;
    }

    const modelId =
      this.configService.get<string>('ai.model') || 'doubao-seed-2.0-pro';
    const prompt =
      'You are preparing a durian analysis task. Review this durian image and briefly summarize what should happen next.';
    this.logger.log(
      `Generating durian AI summary ${JSON.stringify({
        imagePath: input.imagePath ?? null,
        imageUrl: input.imageUrl,
        maxOutputTokens: 120,
        model: modelId,
        prompt,
      })}`,
      'AiService',
    );

    const anthropic = createAnthropic({
      authToken: apiKey,
      baseURL: this.configService.get<string>('ai.baseUrl'),
    });

    const content = await this.buildUserContent(input);
    const result = await generateText({
      model: anthropic(modelId),
      messages: [
        {
          role: 'user',
          content,
        },
      ],
      maxOutputTokens: 120,
    });

    this.logger.log(
      `AI summary generated ${JSON.stringify({
        finishReason: result.finishReason ?? null,
        imagePath: input.imagePath ?? null,
        imageUrl: input.imageUrl,
        model: modelId,
        summary: result.text,
      })}`,
      'AiService',
    );
    return result.text;
  }

  private async buildUserContent(input: {
    imagePath?: string | null;
    imageUrl: string;
  }): Promise<
    Array<
      | { type: 'text'; text: string }
      | { type: 'image'; image: Buffer | string; mediaType?: string }
    >
  > {
    const textPart = {
      type: 'text' as const,
      text: `Image URL for reference: ${input.imageUrl}`,
    };

    if (input.imagePath) {
      const buffer = await readFile(input.imagePath);
      return [
        textPart,
        {
          type: 'image',
          image: buffer,
          mediaType: this.getMediaType(input.imagePath),
        },
      ];
    }

    return [
      textPart,
      {
        type: 'image',
        image: input.imageUrl,
      },
    ];
  }

  private getMediaType(filePath: string): string {
    return (
      IMAGE_MEDIA_TYPE_BY_EXTENSION[extname(filePath).toLowerCase()] ||
      'image/jpeg'
    );
  }
}
