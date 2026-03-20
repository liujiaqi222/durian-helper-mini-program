process.env.NODE_ENV = 'test';

import { join } from 'path';
import { existsSync, rmSync } from 'fs';
import { INestApplication } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import request from 'supertest';
import { App } from 'supertest/types';
import { AppModule } from '../src/app.module';

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
    }).compile();

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

  it('creates an analysis task from a direct file upload', async () => {
    const response = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
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
    expect(['SCORING', 'FAILED']).toContain(body.data.status);
    expect(body.data.taskId).toBeDefined();
  });

  it('rejects analyze requests without a file', async () => {
    const response = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .expect(400);

    expect(response.body.message).toBe('file is required');
  });
});
