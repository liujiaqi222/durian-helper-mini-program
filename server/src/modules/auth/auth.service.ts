import { Injectable, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import jwt from 'jsonwebtoken';
import { UsersService } from '../users/users.service';
import type {
  AuthenticatedUser,
  UserJwtPayload,
  WechatSession,
} from './auth.types';
import { WechatAuthService } from './wechat-auth.service';

@Injectable()
export class AuthService {
  constructor(
    private readonly configService: ConfigService,
    private readonly usersService: UsersService,
    private readonly wechatAuthService: WechatAuthService,
  ) {}

  async login(code: string, inviterCode?: string) {
    const wxSession = await this.wechatAuthService.code2Session(code);
    const { isNewUser, user } = await this.usersService.findOrCreateWechatUser({
      openid: wxSession.openid,
      sessionKey: wxSession.session_key,
      unionid: wxSession.unionid,
    });
    const authedUser =
      isNewUser && inviterCode
        ? await this.usersService.claimInviteReward(user.id, inviterCode)
        : this.usersService.toPublicProfile(user);

    return {
      token: this.sign({ sub: user.publicId }),
      user: authedUser,
    };
  }

  async authenticate(token: string): Promise<AuthenticatedUser> {
    const secret = this.getJwtSecret();
    let payload: UserJwtPayload;

    try {
      payload = jwt.verify(token, secret) as UserJwtPayload;
    } catch {
      throw new UnauthorizedException('invalid token');
    }

    if (!payload?.sub) {
      throw new UnauthorizedException('invalid token');
    }

    const user = await this.usersService.findByPublicId(payload.sub);
    if (!user) {
      throw new UnauthorizedException('user not found');
    }

    return {
      publicId: user.publicId,
      userId: user.id,
    };
  }

  private sign(payload: UserJwtPayload): string {
    const secret = this.getJwtSecret();
    return jwt.sign(payload, secret, { expiresIn: '30d' });
  }

  private getJwtSecret(): string {
    return this.configService.get<string>('jwt.secret') || 'test-jwt-secret';
  }
}
