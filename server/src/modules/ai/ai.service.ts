import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';

@Injectable()
export class AiService {
  constructor(private readonly configService: ConfigService) {}

  async summarizeDurianContext(imageUrl: string): Promise<string> {
    const apiKey = this.configService.get<string>('ai.apiKey');
    if (!apiKey) {
      return `AI not configured for ${imageUrl}`;
    }

    const anthropic = createAnthropic({
      apiKey,
      baseURL: this.configService.get<string>('ai.baseUrl'),
    });

    const result = await generateText({
      model: anthropic(
        this.configService.get<string>('ai.model') || 'doubao-seed-2.0-pro',
      ),
      prompt: `You are preparing a durian analysis task. Summarize what should happen next for this image: ${imageUrl}`,
      maxOutputTokens: 120,
    });

    return result.text;
  }
}
