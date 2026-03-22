import { basename, resolve } from 'path';

type EnvMap = Record<string, string | undefined>;

const DEFAULT_POSTGRES_DB = 'durian_helper';
const DEFAULT_POSTGRES_HOST = '127.0.0.1';
const DEFAULT_POSTGRES_PORT = '5432';
const DEFAULT_POSTGRES_USER = 'postgres';
const DEFAULT_REDIS_DB = '0';
const DEFAULT_REDIS_HOST = '127.0.0.1';
const DEFAULT_REDIS_PORT = '6379';

export function getEnvFilePaths(nodeEnv?: string): string[] {
  const environment = nodeEnv || 'development';
  const serverDir = process.cwd();
  const rootDir =
    basename(serverDir) === 'server' ? resolve(serverDir, '..') : serverDir;

  return [
    resolve(rootDir, `.env.${environment}`),
    resolve(rootDir, '.env'),
  ];
}

export function resolvePostgresUrl(env: EnvMap): string | undefined {
  if (env.POSTGRES_URL?.trim()) {
    return env.POSTGRES_URL.trim();
  }

  const password = env.POSTGRES_PASSWORD?.trim();
  if (!password) {
    return undefined;
  }

  const user = env.POSTGRES_USER?.trim() || DEFAULT_POSTGRES_USER;
  const host = env.POSTGRES_HOST?.trim() || DEFAULT_POSTGRES_HOST;
  const port = env.POSTGRES_PORT?.trim() || DEFAULT_POSTGRES_PORT;
  const database = env.POSTGRES_DB?.trim() || DEFAULT_POSTGRES_DB;

  return `postgresql://${encodeURIComponent(user)}:${encodeURIComponent(password)}@${host}:${port}/${encodeURIComponent(database)}`;
}

export function resolveRedisUrl(env: EnvMap): string | undefined {
  if (env.REDIS_URL?.trim()) {
    return env.REDIS_URL.trim();
  }

  const host = env.REDIS_HOST?.trim();
  if (!host) {
    return undefined;
  }

  const port = env.REDIS_PORT?.trim() || DEFAULT_REDIS_PORT;
  const db = env.REDIS_DB?.trim() || DEFAULT_REDIS_DB;
  const username = env.REDIS_USER?.trim();
  const password = env.REDIS_PASSWORD?.trim();
  const scheme = env.REDIS_TLS?.trim() === 'true' ? 'rediss' : 'redis';

  const auth = username
    ? `${encodeURIComponent(username)}:${encodeURIComponent(password || '')}@`
    : password
      ? `:${encodeURIComponent(password)}@`
      : '';

  return `${scheme}://${auth}${host}:${port}/${db}`;
}

export function hasPostgresConfig(env: EnvMap): boolean {
  return Boolean(resolvePostgresUrl(env));
}

export function hasRedisConfig(env: EnvMap): boolean {
  return Boolean(resolveRedisUrl(env));
}

export const redisDefaults = {
  db: DEFAULT_REDIS_DB,
  host: DEFAULT_REDIS_HOST,
  port: DEFAULT_REDIS_PORT,
};
