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

  it('logs model invocation and output summary', async () => {
    generateTextMock.mockResolvedValue({
      text: 'looks ready for cv analysis',
    } as Awaited<ReturnType<typeof generateText>>);

    const logger = {
      log: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
    } as unknown as LoggerService;
    const config = {
      get: jest.fn((key: string) => {
        switch (key) {
          case 'ai.apiKey':
            return 'test-key';
          case 'ai.baseUrl':
            return 'https://example.com';
          case 'ai.model':
            return 'doubao-seed-2.0-pro';
          default:
            return undefined;
        }
      }),
    } as unknown as ConfigService;

    const service = new AiService(config, logger);
    const result = await service.summarizeDurianContext({
      imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
    });

    expect(result).toBe('looks ready for cv analysis');
    expect((logger.log as jest.Mock).mock.calls).toEqual(
      expect.arrayContaining([
        [expect.stringContaining('Generating durian AI summary'), 'AiService'],
        [expect.stringContaining('AI summary generated'), 'AiService'],
      ]),
    );
  });

  it('uses bearer auth token for Ark requests', async () => {
    generateTextMock.mockResolvedValue({
      text: 'looks ready for cv analysis',
    } as Awaited<ReturnType<typeof generateText>>);

    const logger = {
      log: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
    } as unknown as LoggerService;
    const config = {
      get: jest.fn((key: string) => {
        switch (key) {
          case 'ai.apiKey':
            return 'test-key';
          case 'ai.baseUrl':
            return 'https://ark.cn-beijing.volces.com/api/coding';
          case 'ai.model':
            return 'doubao-seed-2.0-lite';
          default:
            return undefined;
        }
      }),
    } as unknown as ConfigService;

    const service = new AiService(config, logger);
    await service.summarizeDurianContext({
      imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
    });

    expect(createAnthropicMock).toHaveBeenCalledWith({
      authToken: 'test-key',
      baseURL: 'https://ark.cn-beijing.volces.com/api/coding',
    });
  });

  it('sends uploaded local images as multimodal content instead of localhost url text', async () => {
    readFileMock.mockResolvedValue(Buffer.from('fake-image'));
    generateTextMock.mockResolvedValue({
      text: 'ripe durian image received',
    } as Awaited<ReturnType<typeof generateText>>);

    const logger = {
      log: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
    } as unknown as LoggerService;
    const config = {
      get: jest.fn((key: string) => {
        switch (key) {
          case 'ai.apiKey':
            return 'test-key';
          case 'ai.baseUrl':
            return 'https://ark.cn-beijing.volces.com/api/coding';
          case 'ai.model':
            return 'doubao-seed-2.0-lite';
          default:
            return undefined;
        }
      }),
    } as unknown as ConfigService;

    const service = new AiService(config, logger);
    await service.summarizeDurianContext({
      imagePath: '/tmp/uploads/task.jpg',
      imageUrl: 'http://127.0.0.1:3000/uploads/task.jpg',
    });

    expect(readFileMock).toHaveBeenCalledWith('/tmp/uploads/task.jpg');
    expect(generateTextMock).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [
          {
            role: 'user',
            content: [
              expect.objectContaining({
                type: 'text',
              }),
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
});
