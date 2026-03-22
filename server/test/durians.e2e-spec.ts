process.env.NODE_ENV = 'test';

import { join } from 'path';
import { existsSync, rmSync } from 'fs';
import { INestApplication } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import request from 'supertest';
import { App } from 'supertest/types';
import { AppModule } from '../src/app.module';
import { AiService } from '../src/modules/ai/ai.service';
import { CvService } from '../src/modules/durians/cv.service';

describe('Durian analysis endpoints (e2e)', () => {
  let app: INestApplication<App>;
  const uploadsDir = join(process.cwd(), 'uploads');
  const pngPixelBuffer = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aMrcAAAAASUVORK5CYII=',
    'base64',
  );

  beforeEach(async () => {
    if (existsSync(uploadsDir)) {
      rmSync(uploadsDir, { force: true, recursive: true });
    }

    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(CvService)
      .useValue({
        detectAndAnnotate: jest.fn().mockResolvedValue({
          count: 2,
          items: [
            {
              bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
              class_name: 'durian',
              confidence: 0.92,
              cropImageBase64: 'ZmFrZS1jcm9wLUE=',
              label: 'A',
            },
            {
              bbox: { x1: 120, x2: 220, y1: 30, y2: 150 },
              class_name: 'durian',
              confidence: 0.88,
              cropImageBase64: 'ZmFrZS1jcm9wLUI=',
              label: 'B',
            },
          ],
        }),
      })
      .overrideProvider(AiService)
      .useValue({
        scoreDurians: jest.fn().mockResolvedValue({
          overallSummary: 'A 综合表现最好，B 次之。',
          recommendedLabel: 'A',
          items: [
            {
              label: 'A',
              score: 92,
              summary: 'A 更值得买。',
              reasons: ['外形完整', '纹理均匀'],
              risks: ['仅凭图片无法判断内部状态'],
              buyPriority: 1,
            },
            {
              label: 'B',
              score: 84,
              summary: 'B 也可以考虑。',
              reasons: ['成熟度较合适', '体型较饱满'],
              risks: [],
              buyPriority: 2,
            },
          ],
        }),
      })
      .compile();

    app = moduleFixture.createNestApplication();
    app.setGlobalPrefix('api/v1');
    await app.init();
  });

  afterEach(async () => {
    await app.close();
    if (existsSync(uploadsDir)) {
      rmSync(uploadsDir, { force: true, recursive: true });
    }
  });

  async function loginAndGetToken(code: string): Promise<string> {
    const response = await request(app.getHttpServer())
      .post('/api/v1/auth/login')
      .send({ code })
      .expect(201);

    return (response.body as { data: { token: string } }).data.token;
  }

  it('creates an analysis task from a direct file upload', async () => {
    const token = await loginAndGetToken('test-code-durian-a');
    const response = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .set('Authorization', `Bearer ${token}`)
      .attach('file', pngPixelBuffer, {
        contentType: 'image/png',
        filename: 'durian.png',
      })
      .expect(201);

    const body = response.body as {
      code: number;
      data: {
        status: string;
        taskId: string;
      };
    };

    expect(body.code).toBe(0);
    expect(['PENDING', 'DETECTING']).toContain(body.data.status);
    expect(body.data.taskId).toBeDefined();
  });

  it('returns progress data and final scored result', async () => {
    const token = await loginAndGetToken('test-code-durian-b');
    const createResponse = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .set('Authorization', `Bearer ${token}`)
      .attach('file', pngPixelBuffer, {
        contentType: 'image/png',
        filename: 'durian.png',
      })
      .expect(201);

    const taskId = (createResponse.body as { data: { taskId: string } }).data.taskId;

    let taskResponse:
      | request.Response
      | undefined;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      taskResponse = await request(app.getHttpServer())
        .get(`/api/v1/durians/tasks/${taskId}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      if ((taskResponse.body as { data: { status: string } }).data.status === 'DONE') {
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    expect(taskResponse).toBeDefined();
    expect((taskResponse!.body as { data: { detectedCount: number; detectedLabels: string[]; status: string } }).data).toMatchObject({
      detectedCount: 2,
      detectedLabels: ['A', 'B'],
      status: 'DONE',
    });

    const resultResponse = await request(app.getHttpServer())
      .get(`/api/v1/durians/tasks/${taskId}/result`)
      .set('Authorization', `Bearer ${token}`)
      .expect(200);

    expect((resultResponse.body as { data: unknown }).data).toMatchObject({
      sourceImageUrl: expect.stringMatching(
        /^http:\/\/127\.0\.0\.1:\d+\/uploads\/upload-/,
      ),
      overallSummary: 'A 综合表现最好，B 次之。',
      recommendedLabel: 'A',
      items: [
        expect.objectContaining({
          bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
          confidence: 0.92,
          label: 'A',
          score: 92,
        }),
        expect.objectContaining({
          bbox: { x1: 120, x2: 220, y1: 30, y2: 150 },
          confidence: 0.88,
          label: 'B',
          score: 84,
        }),
      ],
    });
  });

  it('rejects analyze requests without a file', async () => {
    const token = await loginAndGetToken('test-code-durian-c');
    const response = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .set('Authorization', `Bearer ${token}`)
      .expect(400);

    expect(response.body.message).toBe('file is required');
  });
});
