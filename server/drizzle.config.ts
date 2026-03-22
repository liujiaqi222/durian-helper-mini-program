import { config as loadEnv } from 'dotenv';
import { defineConfig } from 'drizzle-kit';
import { getEnvFilePaths, resolvePostgresUrl } from './src/config/env';

for (const path of getEnvFilePaths(process.env.NODE_ENV)) {
  loadEnv({ path });
}

export default defineConfig({
  dialect: 'postgresql',
  schema: './src/database/drizzle/schema/index.ts',
  out: './drizzle',
  dbCredentials: {
    url: resolvePostgresUrl(process.env) ?? '',
  },
  strict: true,
  verbose: true,
});
