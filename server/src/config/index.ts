export interface AppConfig {
  environment: string;
  port: number;
  postgres?: string;
  redis?: string;
  publicBaseUrl: string;
  cvService: {
    baseUrl: string;
  };
  ai: {
    apiKey?: string;
    baseUrl: string;
    model: string;
    timeoutMs?: string;
  };
}

export default (): AppConfig => ({
  environment: process.env.NODE_ENV || 'development',
  port: Number(process.env.PORT || 3000),
  postgres: process.env.POSTGRES_URL,
  redis: process.env.REDIS_URL,
  publicBaseUrl:
    process.env.PUBLIC_BASE_URL ||
    `http://127.0.0.1:${Number(process.env.PORT || 3000)}`,
  cvService: {
    baseUrl: process.env.CV_SERVICE_BASE_URL || 'http://127.0.0.1:8010',
  },
  ai: {
    apiKey: process.env.ARK_API_KEY,
    baseUrl:
      process.env.ARK_BASE_URL ||
      'https://ark.cn-beijing.volces.com/api/coding/v1',
    model: process.env.ARK_MODEL || 'doubao-seed-2.0-pro',
    timeoutMs: process.env.ARK_TIMEOUT_MS,
  },
});
