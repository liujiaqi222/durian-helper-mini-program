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

  it('uploads an image and returns both a public url and local path', async () => {
    const response = await request(app.getHttpServer())
      .post('/api/v1/uploads/images')
      .attach('file', pngPixelBuffer, {
        contentType: 'image/png',
        filename: 'durian.png',
      })
      .expect(201);

    const body = response.body as {
      code: number;
      data: {
        fileName: string;
        fileUrl: string;
        localPath: string;
      };
    };

    expect(body.code).toBe(0);
    expect(body.data.fileName).toMatch(/\.png$/);
    expect(body.data.fileUrl).toContain('/uploads/');
    expect(body.data.localPath).toContain('/uploads/');
  });

  it('creates an analysis task from a direct image url', async () => {
    const response = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .send({ imageUrl: 'https://example.com/durian.png' })
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

  it('creates an analysis task from uploaded image metadata', async () => {
    const response = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .send({
        imagePath: '/tmp/uploads/durian.png',
        imageUrl: 'http://127.0.0.1:3000/uploads/durian.png',
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
});
