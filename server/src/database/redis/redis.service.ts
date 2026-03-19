import { Inject, Injectable } from '@nestjs/common';
import { REDIS, type RedisClient } from './redis.provider';

@Injectable()
export class RedisService {
  constructor(@Inject(REDIS) private readonly redis: RedisClient) {}

  async get(key: string): Promise<string | null> {
    if (!this.redis) {
      return null;
    }
    return this.redis.get(key);
  }

  async set(key: string, value: string, ttlSeconds?: number): Promise<void> {
    if (!this.redis) {
      return;
    }

    if (ttlSeconds) {
      await this.redis.setex(key, ttlSeconds, value);
      return;
    }

    await this.redis.set(key, value);
  }

  async del(key: string): Promise<void> {
    if (!this.redis) {
      return;
    }
    await this.redis.del(key);
  }
}
