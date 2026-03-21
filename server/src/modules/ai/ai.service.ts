import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { readFile } from 'fs/promises';
import { extname } from 'path';
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';
import { z } from 'zod';
import { LoggerService } from '../../core/logger/logger.service';
import type { AnalysisBoundingBox } from '../durians/durians.types';

const IMAGE_MEDIA_TYPE_BY_EXTENSION: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
};
const DEFAULT_AI_REQUEST_TIMEOUT_MS = 15_000;

const durianScoreSchema = z.object({
  label: z.string().min(1),
  score: z.number().int().min(0).max(100),
  summary: z.string().min(1),
  reasons: z.array(z.string().min(1)).min(2).max(4),
  risks: z.array(z.string().min(1)).max(3),
});

type DurianScore = z.infer<typeof durianScoreSchema>;

export interface ScoreDurianInput {
  bbox: AnalysisBoundingBox;
  confidence: number;
  cropImageBase64: string | null;
  imagePath?: string | null;
  imageUrl: string;
  label: string;
}

export interface ScoreDurianOutput extends DurianScore {
  buyPriority: number;
}

@Injectable()
export class AiService {
  constructor(
    private readonly configService: ConfigService,
    private readonly logger: LoggerService,
  ) {}

  async scoreDurians(
    items: ScoreDurianInput[],
  ): Promise<{
    overallSummary: string;
    recommendedLabel: string | null;
    items: ScoreDurianOutput[];
  }> {
    if (items.length === 0) {
      return {
        overallSummary: '本次未识别到可评分的榴莲。',
        recommendedLabel: null,
        items: [],
      };
    }

    const apiKey = this.configService.get<string>('ai.apiKey');
    if (!apiKey) {
      return this.buildHeuristicScores(items);
    }

    const modelId =
      this.configService.get<string>('ai.model') || 'doubao-seed-2.0-lite';
    const anthropic = createAnthropic({
      authToken: apiKey,
      baseURL: this.configService.get<string>('ai.baseUrl'),
    });
    const timeoutMs = Number(
      this.configService.get<string>('ai.timeoutMs') ??
        DEFAULT_AI_REQUEST_TIMEOUT_MS,
    );

    const scoredItems: DurianScore[] = [];

    try {
      for (const item of items) {
        this.logger.log(
          `Generating durian score ${JSON.stringify({
            hasCropImage: Boolean(item.cropImageBase64),
            imagePath: item.imagePath ?? null,
            imageUrl: item.imageUrl,
            label: item.label,
            model: modelId,
            timeoutMs,
          })}`,
          'AiService',
        );

        const result = await this.generateScoreWithTimeout(
          {
            model: anthropic(modelId),
            messages: [
              {
                role: 'user',
                content: await this.buildScoringContent(item),
              },
            ],
            maxOutputTokens: 320,
          },
          timeoutMs,
        );
        scoredItems.push(this.parseScoreResult(item.label, result.text));
      }
    } catch (error) {
      this.logger.warn(
        `Falling back to heuristic durian scoring because AI scoring failed ${JSON.stringify({
          error: error instanceof Error ? error.message : 'unknown error',
          labels: items.map((item) => item.label),
        })}`,
        'AiService',
      );
      return this.buildHeuristicScores(items);
    }

    return this.finalizeScores(scoredItems);
  }

  private async buildScoringContent(
    input: ScoreDurianInput,
  ): Promise<
    Array<
      | { type: 'text'; text: string }
      | { type: 'image'; image: Buffer | string; mediaType?: string }
    >
  > {
    const parts: Array<
      | { type: 'text'; text: string }
      | { type: 'image'; image: Buffer | string; mediaType?: string }
    > = [
      {
        type: 'text',
        text: [
          `You are rating durian ${input.label} from a crop image.`,
          'Judge it by visible traits such as color, shape, husk texture, and stem condition.',
          `Bounding box: (${input.bbox.x1}, ${input.bbox.y1}) to (${input.bbox.x2}, ${input.bbox.y2}).`,
          'Return strict JSON with fields: label, score, summary, reasons, risks.',
          'Rules: label must exactly match the requested label.',
          'score must be an integer from 0 to 100.',
          'reasons must contain 2 to 4 short Chinese strings.',
          'risks must contain 0 to 3 short Chinese strings.',
          'summary must be one short Chinese sentence.',
          `Requested label: ${input.label}`,
        ].join(' '),
      },
    ];

    if (input.cropImageBase64) {
      parts.push({
        type: 'image',
        image: Buffer.from(input.cropImageBase64, 'base64'),
        mediaType: 'image/png',
      });
      return parts;
    }

    if (input.imagePath) {
      parts.push({
        type: 'image',
        image: await readFile(input.imagePath),
        mediaType: this.getMediaType(input.imagePath),
      });
    } else {
      parts.push({
        type: 'image',
        image: input.imageUrl,
      });
    }

    return parts;
  }

  private parseScoreResult(expectedLabel: string, text: string): DurianScore {
    const parsed = durianScoreSchema.parse(JSON.parse(text));
    if (parsed.label !== expectedLabel) {
      throw new Error(
        `AI returned mismatched label ${parsed.label} for ${expectedLabel}`,
      );
    }
    return parsed;
  }

  private buildHeuristicScores(
    items: ScoreDurianInput[],
  ): {
    overallSummary: string;
    recommendedLabel: string | null;
    items: ScoreDurianOutput[];
  } {
    this.logger.warn(
      `Falling back to heuristic durian scoring because ai.apiKey is missing ${JSON.stringify({
        labels: items.map((item) => item.label),
      })}`,
      'AiService',
    );

    return this.finalizeScores(
      items.map((item) => ({
        label: item.label,
        score: Math.max(40, Math.min(98, Math.round(item.confidence * 100))),
        summary: `编号 ${item.label} 外观较完整，可作为优先候选。`,
        reasons: ['检测置信度较高', '已完成局部裁剪观察'],
        risks: ['仅凭图片无法判断内部果肉状态'],
      })),
    );
  }

  private finalizeScores(scoredItems: DurianScore[]): {
    overallSummary: string;
    recommendedLabel: string | null;
    items: ScoreDurianOutput[];
  } {
    const ranked = [...scoredItems].sort((a, b) => b.score - a.score);
    const scoredWithPriority = scoredItems.map((item) => ({
      ...item,
      buyPriority:
        ranked.findIndex((candidate) => candidate.label === item.label) + 1,
    }));
    const recommendedLabel = ranked[0]?.label ?? null;

    return {
      overallSummary: recommendedLabel
        ? `本次共识别出 ${scoredItems.length} 个榴莲，${recommendedLabel} 综合表现最好，建议优先选择。`
        : '本次未形成明确推荐结果。',
      recommendedLabel,
      items: scoredWithPriority,
    };
  }

  private getMediaType(filePath: string): string {
    return (
      IMAGE_MEDIA_TYPE_BY_EXTENSION[extname(filePath).toLowerCase()] ||
      'image/jpeg'
    );
  }

  private async generateScoreWithTimeout(
    input: Parameters<typeof generateText>[0],
    timeoutMs: number,
  ): Promise<Awaited<ReturnType<typeof generateText>>> {
    return Promise.race([
      generateText(input),
      new Promise<never>((_, reject) => {
        setTimeout(() => {
          reject(new Error(`AI scoring timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      }),
    ]);
  }
}
