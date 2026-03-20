import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';
import { LoggerService } from '../../core/logger/logger.service';

@Injectable()
export class AiService {
  constructor(
    private readonly configService: ConfigService,
    private readonly logger: LoggerService,
  ) {}

  async summarizeDurianContext(imageUrl: string): Promise<string> {
    const apiKey = this.configService.get<string>('ai.apiKey');
    if (!apiKey) {
      this.logger.warn(
        `Skipping durian AI summary because ai.apiKey is missing ${JSON.stringify({
          imageUrl,
        })}`,
        'AiService',
      );
      return `AI not configured for ${imageUrl}`;
    }

    const modelId =
      this.configService.get<string>('ai.model') || 'doubao-seed-2.0-pro';
    const prompt = `You are preparing a durian analysis task. Summarize what should happen next for this image: ${imageUrl}`;
    this.logger.log(
      `Generating durian AI summary ${JSON.stringify({
        imageUrl,
        maxOutputTokens: 120,
        model: modelId,
        prompt,
      })}`,
      'AiService',
    );

    const anthropic = createAnthropic({
      apiKey,
      baseURL: this.configService.get<string>('ai.baseUrl'),
    });

    const result = await generateText({
      model: anthropic(modelId),
      prompt,
      maxOutputTokens: 120,
    });

    this.logger.log(
      `AI summary generated ${JSON.stringify({
        finishReason: result.finishReason ?? null,
        imageUrl,
        model: modelId,
        summary: result.text,
      })}`,
      'AiService',
    );
    return result.text;
  }
}
