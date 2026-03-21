import { ConfigService } from '@nestjs/config';
import { readFile } from 'fs/promises';
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';
import { LoggerService } from '../../core/logger/logger.service';
import { AiService } from './ai.service';

jest.mock('@ai-sdk/anthropic', () => ({
  createAnthropic: jest.fn(),
}));

jest.mock('ai', () => ({
  generateText: jest.fn(),
}));

jest.mock('fs/promises', () => ({
  readFile: jest.fn(),
}));

describe('AiService', () => {
  const createAnthropicMock = createAnthropic as jest.MockedFunction<
    typeof createAnthropic
  >;
  const generateTextMock = generateText as jest.MockedFunction<
    typeof generateText
  >;
  const readFileMock = readFile as jest.MockedFunction<typeof readFile>;

  beforeEach(() => {
    createAnthropicMock.mockReset();
    generateTextMock.mockReset();
    readFileMock.mockReset();
    createAnthropicMock.mockReturnValue(
      jest.fn() as ReturnType<typeof createAnthropic>,
    );
  });

  function createService(configOverrides?: Record<string, string | undefined>) {
    const logger = {
      log: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
    } as unknown as LoggerService;
    const config = {
      get: jest.fn((key: string) => {
        const defaults: Record<string, string | undefined> = {
          'ai.apiKey': 'test-key',
          'ai.baseUrl': 'https://example.com',
          'ai.model': 'doubao-seed-2.0-pro',
        };

        return { ...defaults, ...configOverrides }[key];
      }),
    } as unknown as ConfigService;

    return {
      logger,
      service: new AiService(config, logger),
    };
  }

  it('parses strict JSON scoring results and ranks buy priority', async () => {
    generateTextMock.mockResolvedValueOnce({
      text: JSON.stringify({
        label: 'A',
        score: 91,
        summary: '编号 A 更适合买。',
        reasons: ['外形完整', '果刺分布均匀'],
        risks: ['仅凭图片无法判断内部状态'],
      }),
    } as Awaited<ReturnType<typeof generateText>>);

    const { service, logger } = createService();
    const result = await service.scoreDurians([
      {
        bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
        confidence: 0.91,
        cropImageBase64: 'ZmFrZS1jcm9w',
        imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
        label: 'A',
      },
    ]);

    expect(result.recommendedLabel).toBe('A');
    expect(result.items[0].buyPriority).toBe(1);
    expect(result.overallSummary).toContain('A');
    expect((logger.log as jest.Mock).mock.calls).toEqual(
      expect.arrayContaining([
        [expect.stringContaining('Generating durian score'), 'AiService'],
      ]),
    );
  });

  it('uses bearer auth token for Ark requests', async () => {
    generateTextMock.mockResolvedValue({
      text: JSON.stringify({
        label: 'A',
        score: 88,
        summary: '编号 A 可购买。',
        reasons: ['果形较饱满', '成熟度较合适'],
        risks: [],
      }),
    } as Awaited<ReturnType<typeof generateText>>);

    const { service } = createService({
      'ai.baseUrl': 'https://ark.cn-beijing.volces.com/api/coding',
      'ai.model': 'doubao-seed-2.0-lite',
    });
    await service.scoreDurians([
      {
        bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
        confidence: 0.88,
        cropImageBase64: 'ZmFrZS1jcm9w',
        imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
        label: 'A',
      },
    ]);

    expect(createAnthropicMock).toHaveBeenCalledWith({
      authToken: 'test-key',
      baseURL: 'https://ark.cn-beijing.volces.com/api/coding',
    });
  });

  it('sends uploaded local images and crop images as multimodal content', async () => {
    readFileMock.mockResolvedValue(Buffer.from('fake-image'));
    generateTextMock.mockResolvedValue({
      text: JSON.stringify({
        label: 'A',
        score: 90,
        summary: '编号 A 外观较好。',
        reasons: ['果形完整', '纹理清晰'],
        risks: ['仅凭图片无法判断内部状态'],
      }),
    } as Awaited<ReturnType<typeof generateText>>);

    const { service } = createService();
    await service.scoreDurians([
      {
        bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
        confidence: 0.9,
        cropImageBase64: 'ZmFrZS1jcm9w',
        imagePath: '/tmp/uploads/task.jpg',
        imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
        label: 'A',
      },
    ]);

    expect(readFileMock).not.toHaveBeenCalled();
    expect(generateTextMock).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [
          {
            role: 'user',
            content: [
              expect.objectContaining({ type: 'text' }),
              {
                type: 'image',
                image: Buffer.from('fake-crop'),
                mediaType: 'image/png',
              },
            ],
          },
        ],
      }),
    );
  });

  it('falls back to heuristic scoring when ai api key is missing', async () => {
    const { service, logger } = createService({
      'ai.apiKey': undefined,
    });

    const result = await service.scoreDurians([
      {
        bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
        confidence: 0.82,
        cropImageBase64: null,
        imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
        label: 'A',
      },
    ]);

    expect(result.items[0].score).toBe(82);
    expect(result.recommendedLabel).toBe('A');
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('heuristic durian scoring'),
      'AiService',
    );
  });

  it('falls back to the source image when crop image is missing', async () => {
    readFileMock.mockResolvedValue(Buffer.from('fake-image'));
    generateTextMock.mockResolvedValue({
      text: JSON.stringify({
        label: 'A',
        score: 85,
        summary: '编号 A 可购买。',
        reasons: ['果形较饱满', '纹理较清晰'],
        risks: [],
      }),
    } as Awaited<ReturnType<typeof generateText>>);

    const { service } = createService();
    await service.scoreDurians([
      {
        bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
        confidence: 0.85,
        cropImageBase64: null,
        imagePath: '/tmp/uploads/task.jpg',
        imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
        label: 'A',
      },
    ]);

    expect(readFileMock).toHaveBeenCalledWith('/tmp/uploads/task.jpg');
    expect(generateTextMock).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [
          {
            role: 'user',
            content: [
              expect.objectContaining({ type: 'text' }),
              {
                type: 'image',
                image: Buffer.from('fake-image'),
                mediaType: 'image/jpeg',
              },
            ],
          },
        ],
      }),
    );
  });

  it('falls back to heuristic scoring when ai generation times out', async () => {
    jest.useFakeTimers();
    generateTextMock.mockImplementation(
      () =>
        new Promise(() => {
          // Intentionally unresolved to simulate a hanging model call.
        }) as ReturnType<typeof generateText>,
    );

    const { service, logger } = createService({
      'ai.timeoutMs': '5',
    });

    const resultPromise = service.scoreDurians([
      {
        bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
        confidence: 0.82,
        cropImageBase64: 'ZmFrZS1jcm9w',
        imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
        label: 'A',
      },
    ]);

    await jest.advanceTimersByTimeAsync(5);
    const result = await resultPromise;

    expect(result.items[0].score).toBe(82);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('AI scoring failed'),
      'AiService',
    );

    jest.useRealTimers();
  });
});
