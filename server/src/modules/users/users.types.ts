export type CreditTransactionType =
  | 'INITIAL_GRANT'
  | 'ANALYZE_CONSUME'
  | 'AD_REWARD'
  | 'INVITE_REWARD'
  | 'INVITEE_REWARD';

export interface AppUser {
  id: number;
  publicId: string;
  openid: string;
  unionid: string | null;
  sessionKey: string | null;
  name: string | null;
  phone: string | null;
  remainingCredits: number;
  usedCredits: number;
  inviteCode: string;
  invitedByUserId: number | null;
  adRewardCount: number;
  inviteRewardCount: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface PublicUserProfile {
  publicId: string;
  name: string | null;
  phone: string | null;
  remainingCredits: number;
  usedCredits: number;
  inviteCode: string;
  adRewardCount: number;
  inviteRewardCount: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface CreateWechatUserInput {
  openid: string;
  unionid?: string | null;
  sessionKey?: string | null;
}

export interface UpdateWechatSessionInput {
  userId: number;
  sessionKey?: string | null;
  unionid?: string | null;
}

export interface UsersRepository {
  createWechatUser(input: CreateWechatUserInput): Promise<AppUser>;
  findByInviteCode(inviteCode: string): Promise<AppUser | null>;
  findByOpenidOrUnionid(
    openid: string,
    unionid?: string | null,
  ): Promise<AppUser | null>;
  findByPublicId(publicId: string): Promise<AppUser | null>;
  findByUserId(userId: number): Promise<AppUser | null>;
  grantAdReward(userId: number): Promise<AppUser>;
  bindInviter(input: {
    userId: number;
    inviterCode: string;
    inviteeRewardCredits: number;
    inviterRewardCredits: number;
  }): Promise<AppUser>;
  consumeCredit(input: { userId: number; reason: string }): Promise<AppUser>;
  updateWechatSession(input: UpdateWechatSessionInput): Promise<AppUser>;
}
