import {
  ArgumentsHost,
  Catch,
  ExceptionFilter,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import type { Request, Response } from 'express';
import { LoggerService } from '../logger/logger.service';

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  constructor(private readonly logger: LoggerService) {}

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    const status =
      exception instanceof HttpException
        ? exception.getStatus()
        : HttpStatus.INTERNAL_SERVER_ERROR;

    const exceptionResponse =
      exception instanceof HttpException ? exception.getResponse() : null;

    let message = 'Internal server error';
    if (typeof exceptionResponse === 'string') {
      message = exceptionResponse;
    } else if (
      typeof exceptionResponse === 'object' &&
      exceptionResponse !== null &&
      'message' in exceptionResponse
    ) {
      const detail = (exceptionResponse as { message?: string | string[] })
        .message;
      message = Array.isArray(detail) ? detail.join(', ') : String(detail);
    } else if (exception instanceof Error) {
      message = exception.message;
    }

    if (status >= 500) {
      this.logger.error(
        {
          body: request.body as Record<string, unknown>,
          method: request.method,
          params: request.params,
          query: request.query,
          url: request.url,
        },
        exception instanceof Error ? exception.stack : String(exception),
        'AllExceptionsFilter',
      );
    } else {
      this.logger.warn(
        `${request.method} ${request.url} -> ${status} ${message}`,
        'AllExceptionsFilter',
      );
    }

    response.status(status).json({
      code: status,
      message,
      path: request.url,
      timestamp: new Date().toISOString(),
    });
  }
}
