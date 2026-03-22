process.env.NODE_ENV = 'test';

import { INestApplication } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import request from 'supertest';
import { App } from 'supertest/types';
import { AppModule } from '../src/app.module';
import { AiService } from '../src/modules/ai/ai.service';
import { CvService } from '../src/modules/durians/cv.service';

describe('Auth and credits endpoints (e2e)', () => {
  let app: INestApplication<App>;
  const pngPixelBuffer = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aMrcAAAAASUVORK5CYII=',
    'base64',
  );

  beforeEach(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(CvService)
      .useValue({
        detectAndAnnotate: jest.fn().mockResolvedValue({
          count: 1,
          items: [
            {
              bbox: { x1: 10, x2: 100, y1: 20, y2: 120 },
              class_name: 'durian',
              confidence: 0.92,
              cropImageBase64: 'ZmFrZS1jcm9wLUE=',
              label: 'A',
            },
          ],
        }),
      })
      .overrideProvider(AiService)
      .useValue({
        scoreDurians: jest.fn().mockResolvedValue({
          overallSummary: 'A 综合表现最好。',
          recommendedLabel: 'A',
          items: [
            {
              label: 'A',
              score: 92,
              summary: 'A 更值得买。',
              reasons: ['外形完整'],
              risks: ['仅凭图片无法判断内部状态'],
              buyPriority: 1,
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
  });

  it('logs in, grants initial credits, rewards invited users during login, and consumes credits on analyze', async () => {
    const inviterLoginResponse = await request(app.getHttpServer())
      .post('/api/v1/auth/login')
      .send({ code: 'test-code-user-a' })
      .expect(201);

    const inviterLoginBody = inviterLoginResponse.body as {
      data: {
        token: string;
        user: {
          inviteCode: string;
          publicId: string;
          remainingCredits: number;
        };
      };
    };

    expect(inviterLoginBody.data.token).toEqual(expect.any(String));
    expect(inviterLoginBody.data.user.publicId).toEqual(expect.any(String));
    expect(inviterLoginBody.data.user.inviteCode).toEqual(expect.any(String));
    expect(inviterLoginBody.data.user.remainingCredits).toBe(3);

    const inviterToken = inviterLoginBody.data.token;
    const inviterCode = inviterLoginBody.data.user.inviteCode;

    const meResponse = await request(app.getHttpServer())
      .get('/api/v1/users/me')
      .set('Authorization', `Bearer ${inviterToken}`)
      .expect(200);

    expect((meResponse.body as { data: { remainingCredits: number } }).data.remainingCredits).toBe(3);

    const inviteeLoginResponse = await request(app.getHttpServer())
      .post('/api/v1/auth/login')
      .send({
        code: 'test-code-user-b',
        inviterCode,
      })
      .expect(201);

    const inviteeLoginBody = inviteeLoginResponse.body as {
      data: {
        token: string;
        user: {
          remainingCredits: number;
        };
      };
    };
    const inviteeToken = inviteeLoginBody.data.token;

    expect(inviteeLoginBody.data.user.remainingCredits).toBe(4);

    const inviterAfterInviteResponse = await request(app.getHttpServer())
      .get('/api/v1/users/me')
      .set('Authorization', `Bearer ${inviterToken}`)
      .expect(200);

    expect((inviterAfterInviteResponse.body as { data: { remainingCredits: number } }).data.remainingCredits).toBe(5);

    const adRewardResponse = await request(app.getHttpServer())
      .post('/api/v1/users/me/rewards/ad')
      .set('Authorization', `Bearer ${inviterToken}`)
      .send({})
      .expect(201);

    expect((adRewardResponse.body as { data: { remainingCredits: number } }).data.remainingCredits).toBe(6);

    await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .attach('file', pngPixelBuffer, {
        contentType: 'image/png',
        filename: 'durian.png',
      })
      .expect(401);

    const analyzeResponse = await request(app.getHttpServer())
      .post('/api/v1/durians/analyze')
      .set('Authorization', `Bearer ${inviterToken}`)
      .attach('file', pngPixelBuffer, {
        contentType: 'image/png',
        filename: 'durian.png',
      })
      .expect(201);

    expect((analyzeResponse.body as { data: { taskId: string } }).data.taskId).toEqual(
      expect.any(String),
    );

    const afterAnalyzeMeResponse = await request(app.getHttpServer())
      .get('/api/v1/users/me')
      .set('Authorization', `Bearer ${inviterToken}`)
      .expect(200);

    expect((afterAnalyzeMeResponse.body as { data: { remainingCredits: number; usedCredits: number } }).data).toMatchObject({
      remainingCredits: 5,
      usedCredits: 1,
    });
  });

  it('does not expose the manual invite-claim endpoint anymore', async () => {
    const loginResponse = await request(app.getHttpServer())
      .post('/api/v1/auth/login')
      .send({ code: 'test-code-user-c' })
      .expect(201);

    const token = (loginResponse.body as { data: { token: string } }).data.token;

    await request(app.getHttpServer())
      .post('/api/v1/users/me/rewards/invite')
      .set('Authorization', `Bearer ${token}`)
      .send({ inviterCode: 'INVABC1' })
      .expect(404);
  });
});
