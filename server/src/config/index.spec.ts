import config from './index';

describe('config', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env.ARK_BASE_URL;
    delete process.env.POSTGRES_URL;
    delete process.env.POSTGRES_DB;
    delete process.env.POSTGRES_HOST;
    delete process.env.POSTGRES_PASSWORD;
    delete process.env.POSTGRES_PORT;
    delete process.env.POSTGRES_USER;
    delete process.env.REDIS_URL;
    delete process.env.REDIS_DB;
    delete process.env.REDIS_HOST;
    delete process.env.REDIS_PASSWORD;
    delete process.env.REDIS_PORT;
    delete process.env.REDIS_TLS;
    delete process.env.REDIS_USER;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('defaults Ark base URL to the Claude-compatible v1 prefix', () => {
    const value = config();

    expect(value.ai.baseUrl).toBe(
      'https://ark.cn-beijing.volces.com/api/coding/v1',
    );
  });

  it('derives Postgres and Redis URLs from component fields', () => {
    process.env.POSTGRES_HOST = '127.0.0.1';
    process.env.POSTGRES_PORT = '5432';
    process.env.POSTGRES_DB = 'durian_helper';
    process.env.POSTGRES_USER = 'durian_admin';
    process.env.POSTGRES_PASSWORD = 'secret/with:symbols';
    process.env.REDIS_HOST = '127.0.0.1';
    process.env.REDIS_PORT = '6379';
    process.env.REDIS_DB = '1';
    process.env.REDIS_TLS = 'true';
    process.env.REDIS_PASSWORD = 'redis-pass';

    const value = config();

    expect(value.postgres).toBe(
      'postgresql://durian_admin:secret%2Fwith%3Asymbols@127.0.0.1:5432/durian_helper',
    );
    expect(value.redis).toBe('rediss://:redis-pass@127.0.0.1:6379/1');
  });
});
