import { Global, Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { APP_FILTER, APP_INTERCEPTOR } from '@nestjs/core';
import config from '../config';
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
      envFilePath: ['.env', `.env.${process.env.NODE_ENV || 'development'}`],
      validate: (env: Record<string, string | undefined>) => {
        const isTest = (env.NODE_ENV || 'development') === 'test';
        if (!isTest && !env.POSTGRES_URL?.trim()) {
          throw new Error('Missing required env var: POSTGRES_URL');
        }
        if (!isTest && !env.REDIS_URL?.trim()) {
          throw new Error('Missing required env var: REDIS_URL');
        }
        if (!isTest && !env.ARK_API_KEY?.trim()) {
          throw new Error('Missing required env var: ARK_API_KEY');
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
