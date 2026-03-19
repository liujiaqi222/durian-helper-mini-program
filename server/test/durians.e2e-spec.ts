process.env.NODE_ENV = 'test';

import { INestApplication } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import request from 'supertest';
import { App } from 'supertest/types';
import { AppModule } from '../src/app.module';

describe('Durian analysis endpoints (e2e)', () => {
  let app: INestApplication<App>;

  beforeEach(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.setGlobalPrefix('api/v1');
    await app.init();
  });

  afterEach(async () => {
    await app.close();
  });

  it('creates an analysis task', async () => {
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
    expect(body.data.status).toBe('PENDING');
    expect(body.data.taskId).toBeDefined();
  });
});
