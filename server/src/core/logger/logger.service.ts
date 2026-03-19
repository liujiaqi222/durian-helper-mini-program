import { Injectable, LoggerService as NestLogger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as winston from 'winston';
import DailyRotateFile from 'winston-daily-rotate-file';

@Injectable()
export class LoggerService implements NestLogger {
  private readonly logger: winston.Logger;

  constructor(private readonly configService: ConfigService) {
    const isDevelopment =
      this.configService.get<string>('environment') === 'development';

    this.logger = winston.createLogger({
      level: isDevelopment ? 'debug' : 'info',
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize({ all: true }),
            winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
            winston.format.printf((info) => {
              const context =
                typeof info.context === 'string' ? ` [${info.context}]` : '';
              const trace =
                typeof info.trace === 'string' ? `\n${info.trace}` : '';
              const message = this.stringifyMessage(info.message);
              const timestamp = this.stringifyMessage(info.timestamp);
              const level = this.stringifyMessage(info.level);
              return `[Nest] ${timestamp} ${level}${context} ${message}${trace}`;
            }),
          ),
        }),
        new DailyRotateFile({
          dirname: 'logs',
          filename: 'server-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          maxFiles: '14d',
          maxSize: '20m',
          zippedArchive: true,
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json(),
          ),
        }),
      ],
    });
  }

  log(message: unknown, context?: string) {
    this.logger.info(this.stringifyMessage(message), { context });
  }

  error(message: unknown, trace?: string, context?: string) {
    this.logger.error(this.stringifyMessage(message), { context, trace });
  }

  warn(message: unknown, context?: string) {
    this.logger.warn(this.stringifyMessage(message), { context });
  }

  debug(message: unknown, context?: string) {
    this.logger.debug(this.stringifyMessage(message), { context });
  }

  verbose(message: unknown, context?: string) {
    this.logger.verbose(this.stringifyMessage(message), { context });
  }

  private stringifyMessage(message: unknown): string {
    if (typeof message === 'string') {
      return message;
    }
    if (message === undefined) {
      return 'undefined';
    }
    return JSON.stringify(message);
  }
}
