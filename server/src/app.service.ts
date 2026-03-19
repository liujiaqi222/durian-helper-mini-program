import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getHealth() {
    return {
      service: 'durian-helper-server',
      status: 'ok',
    };
  }
}
