import { config as loadEnv } from 'dotenv';
import { defineConfig } from 'drizzle-kit';

loadEnv({
  path: `.env.${process.env.NODE_ENV || 'development'}`,
});
loadEnv();

export default defineConfig({
  dialect: 'postgresql',
  schema: './src/database/drizzle/schema/index.ts',
  out: './drizzle',
  dbCredentials: {
    url: process.env.POSTGRES_URL ?? '',
  },
  strict: true,
  verbose: true,
});
