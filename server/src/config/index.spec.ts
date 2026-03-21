import config from './index';

describe('config', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env.ARK_BASE_URL;
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
});
