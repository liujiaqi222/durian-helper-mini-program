import { Global, Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { APP_FILTER, APP_INTERCEPTOR } from '@nestjs/core';
import config from '../config';
import { getEnvFilePaths, hasPostgresConfig, hasRedisConfig } from '../config/env';
import { DrizzleModule } from '../database/drizzle/drizzle.module';
import { RedisModule } from '../database/redis/redis.module';
import { AllExceptionsFilter } from './all-exceptions/all-exceptions.filter';
import { TransformResponseInterceptor } from './interceptors/transform-response/transform-response.interceptor';
import { LoggerService } from './logger/logger.service';

@Global()
@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      load: [config],
      envFilePath: getEnvFilePaths(process.env.NODE_ENV),
      validate: (env: Record<string, string | undefined>) => {
        const isTest = (env.NODE_ENV || 'development') === 'test';
        if (!isTest && !hasPostgresConfig(env)) {
          throw new Error(
            'Missing required Postgres config: set POSTGRES_URL or POSTGRES_PASSWORD with host/user/db fields',
          );
        }
        if (!isTest && !hasRedisConfig(env)) {
          throw new Error(
            'Missing required Redis config: set REDIS_URL or REDIS_HOST',
          );
        }
        if (!isTest && !env.ARK_API_KEY?.trim()) {
          throw new Error('Missing required env var: ARK_API_KEY');
        }
        if (!isTest && !env.JWT_SECRET?.trim()) {
          throw new Error('Missing required env var: JWT_SECRET');
        }
        return env;
      },
    }),
    RedisModule,
    DrizzleModule,
  ],
  providers: [
    LoggerService,
    { provide: APP_INTERCEPTOR, useClass: TransformResponseInterceptor },
    { provide: APP_FILTER, useClass: AllExceptionsFilter },
  ],
  exports: [LoggerService, DrizzleModule, RedisModule],
})
export class CoreModule {}
