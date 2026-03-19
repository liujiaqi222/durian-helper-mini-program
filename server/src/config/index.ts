export interface AppConfig {
  environment: string;
  port: number;
  postgres?: string;
  redis?: string;
  ai: {
    apiKey?: string;
    baseUrl: string;
    model: string;
  };
}

export default (): AppConfig => ({
  environment: process.env.NODE_ENV || 'development',
  port: Number(process.env.PORT || 3000),
  postgres: process.env.POSTGRES_URL,
  redis: process.env.REDIS_URL,
  ai: {
    apiKey: process.env.ARK_API_KEY,
    baseUrl:
      process.env.ARK_BASE_URL ||
      'https://ark.cn-beijing.volces.com/api/coding',
    model: process.env.ARK_MODEL || 'doubao-seed-2.0-pro',
  },
});
