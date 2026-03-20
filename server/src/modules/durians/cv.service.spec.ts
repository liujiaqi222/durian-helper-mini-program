import { ConfigService } from '@nestjs/config';
import { LoggerService } from '../../core/logger/logger.service';
import { UploadsService } from '../uploads/uploads.service';
import { CvService } from './cv.service';

describe('CvService', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.restoreAllMocks();
  });

  it('logs cv request and response summary', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        annotated_image_base64: 'ZmFrZQ==',
        count: 1,
        items: [
          {
            bbox: { x1: 1, x2: 2, y1: 3, y2: 4 },
            class_name: 'durian',
            confidence: 0.98,
            crop_image_base64: 'Y3JvcA==',
            label: 'A',
          },
        ],
        message: 'ok',
      }),
    }) as typeof fetch;

    const uploadsService = {
      storeBase64Image: jest
        .fn()
        .mockResolvedValueOnce({
          fileUrl: 'http://127.0.0.1:3000/uploads/task_1-annotated.jpg',
        })
        .mockResolvedValueOnce({
          fileUrl: 'http://127.0.0.1:3000/uploads/task_1-A.jpg',
        }),
    } as unknown as UploadsService;
    const config = {
      get: jest.fn((key: string) =>
        key === 'cvService.baseUrl' ? 'http://127.0.0.1:8010' : undefined,
      ),
    } as unknown as ConfigService;
    const logger = {
      log: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
    } as unknown as LoggerService;

    const service = new CvService(config, uploadsService, logger);
    const result = await service.detectAndAnnotate({
      imageUrl: 'http://127.0.0.1:3000/uploads/task_1.jpg',
      taskId: 'task_1',
    });

    expect(result.annotatedImageUrl).toBe(
      'http://127.0.0.1:3000/uploads/task_1-annotated.jpg',
    );
    expect((logger.log as jest.Mock).mock.calls).toEqual(
      expect.arrayContaining([
        [
          expect.stringContaining('Calling cv-service detect-and-annotate'),
          'CvService',
        ],
        [
          expect.stringContaining('cv-service detect-and-annotate completed'),
          'CvService',
        ],
      ]),
    );
  });
});
