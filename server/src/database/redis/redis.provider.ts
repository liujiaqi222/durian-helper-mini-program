import { Provider } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis from 'ioredis';
import { LoggerService } from '../../core/logger/logger.service';

export const REDIS = 'REDIS';

export type RedisClient = Pick<Redis, 'get' | 'set' | 'setex' | 'del'> | null;

export const RedisProvider: Provider = {
  provide: REDIS,
  inject: [ConfigService, LoggerService],
  useFactory: (
    configService: ConfigService,
    logger: LoggerService,
  ): RedisClient => {
    if (configService.get<string>('environment') === 'test') {
      return null;
    }

    const redisUrl = configService.get<string>('redis');
    if (!redisUrl) {
      throw new Error('REDIS_URL is not configured');
    }

    const client = new Redis(redisUrl);
    client.on('connect', () => logger.log('Redis ready', 'RedisModule'));
    client.on('error', (error) =>
      logger.error(error.message, error.stack, 'RedisModule'),
    );

    return client;
  },
};
