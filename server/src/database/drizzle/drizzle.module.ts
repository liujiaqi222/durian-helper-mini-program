import { Module } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { drizzle, type NodePgDatabase } from 'drizzle-orm/node-postgres';
import { Pool } from 'pg';
import { LoggerService } from '../../core/logger/logger.service';
import * as schema from './schema';

export const DRIZZLE = Symbol('DRIZZLE');
export type DrizzleDb = NodePgDatabase<typeof schema>;

@Module({
  providers: [
    {
      provide: DRIZZLE,
      inject: [ConfigService, LoggerService],
      useFactory: async (
        configService: ConfigService,
        logger: LoggerService,
      ): Promise<DrizzleDb | null> => {
        if (configService.get<string>('environment') === 'test') {
          return null;
        }

        const connectionString = configService.get<string>('postgres');
        if (!connectionString) {
          throw new Error('POSTGRES_URL is not configured');
        }

        const pool = new Pool({
          connectionString,
          max: 10,
          ssl: connectionString.includes('sslmode=require')
            ? { rejectUnauthorized: false }
            : undefined,
        });

        pool.on('error', (error) => {
          logger.error(error.message, error.stack, 'DrizzleModule');
        });

        const client = await pool.connect();
        client.release();
        logger.log('Postgres pool ready', 'DrizzleModule');

        return drizzle(pool, { schema });
      },
    },
  ],
  exports: [DRIZZLE],
})
export class DrizzleModule {}
