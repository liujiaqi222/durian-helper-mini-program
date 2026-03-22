export interface AuthenticatedUser {
  userId: number;
  publicId: string;
}

export interface UserJwtPayload {
  sub: string;
}

export interface WechatSession {
  openid: string;
  session_key: string;
  unionid?: string;
}
