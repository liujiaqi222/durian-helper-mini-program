import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { WechatSession } from './auth.types';

@Injectable()
export class WechatAuthService {
  constructor(private readonly configService: ConfigService) {}

  async code2Session(code: string): Promise<WechatSession> {

    const appId = this.configService.get<string>('wechat.appId');
    const secret = this.configService.get<string>('wechat.appSecret');
    if (!appId || !secret) {
      throw new ServiceUnavailableException(
        'wechat app credentials are not configured',
      );
    }

    const url = new URL('https://api.weixin.qq.com/sns/jscode2session');
    url.searchParams.set('appid', appId);
    url.searchParams.set('secret', secret);
    url.searchParams.set('js_code', code);
    url.searchParams.set('grant_type', 'authorization_code');

    const response = await fetch(url);
    const payload = (await response.json()) as
      | WechatSession
      | { errcode?: number; errmsg?: string };

    if (!response.ok || 'errcode' in payload || !('openid' in payload)) {
      throw new ServiceUnavailableException(
        ('errmsg' in payload && payload.errmsg) || 'wechat login failed',
      );
    }

    return payload;
  }
}
