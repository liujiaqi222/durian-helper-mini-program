import { ConfigService } from '@nestjs/config';
import { generateText } from 'ai';
import { LoggerService } from '../../core/logger/logger.service';
import { AiService } from './ai.service';

jest.mock('ai', () => ({
  generateText: jest.fn(),
}));

describe('AiService', () => {
  const generateTextMock = generateText as jest.MockedFunction<typeof generateText>;

  beforeEach(() => {
    generateTextMock.mockReset();
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
    const result = await service.summarizeDurianContext(
      'http://127.0.0.1:3000/uploads/task.jpg',
    );

    expect(result).toBe('looks ready for cv analysis');
    expect((logger.log as jest.Mock).mock.calls).toEqual(
      expect.arrayContaining([
        [
          expect.stringContaining('Generating durian AI summary'),
          'AiService',
        ],
        [
          expect.stringContaining('AI summary generated'),
          'AiService',
        ],
      ]),
    );
  });
});
