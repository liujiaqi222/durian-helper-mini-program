import {
  BadRequestException,
  Inject,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import {
  AD_REWARD_CREDITS,
  INVITEE_REWARD_CREDITS,
  INVITER_REWARD_CREDITS,
  USERS_REPOSITORY,
} from './users.constants';
import type {
  AppUser,
  PublicUserProfile,
  UsersRepository,
} from './users.types';

@Injectable()
export class UsersService {
  constructor(
    @Inject(USERS_REPOSITORY)
    private readonly repository: UsersRepository,
  ) {}

  async findByPublicId(publicId: string): Promise<AppUser | null> {
    return this.repository.findByPublicId(publicId);
  }

  async getProfile(userId: number): Promise<PublicUserProfile> {
    const user = await this.repository.findByUserId(userId);
    if (!user) {
      throw new NotFoundException('user not found');
    }
    return this.toPublicProfile(user);
  }

  async findOrCreateWechatUser(input: {
    openid: string;
    unionid?: string | null;
    sessionKey?: string | null;
  }): Promise<{ user: AppUser; isNewUser: boolean }> {
    const existing = await this.repository.findByOpenidOrUnionid(
      input.openid,
      input.unionid,
    );

    if (!existing) {
      return {
        isNewUser: true,
        user: await this.repository.createWechatUser(input),
      };
    }

    return {
      isNewUser: false,
      user: await this.repository.updateWechatSession({
        userId: existing.id,
        sessionKey: input.sessionKey,
        unionid: input.unionid,
      }),
    };
  }

  async grantAdReward(userId: number): Promise<PublicUserProfile> {
    const user = await this.repository.grantAdReward(userId);
    return this.toPublicProfile(user);
  }

  async claimInviteReward(
    userId: number,
    inviterCode: string,
  ): Promise<PublicUserProfile> {
    if (!inviterCode.trim()) {
      throw new BadRequestException('inviterCode is required');
    }

    const user = await this.repository.bindInviter({
      userId,
      inviterCode: inviterCode.trim(),
      inviteeRewardCredits: INVITEE_REWARD_CREDITS,
      inviterRewardCredits: INVITER_REWARD_CREDITS,
    });

    return this.toPublicProfile(user);
  }

  async consumeAnalyzeCredit(userId: number): Promise<PublicUserProfile> {
    const user = await this.repository.consumeCredit({
      userId,
      reason: 'durian-analyze',
    });
    return this.toPublicProfile(user);
  }

  toPublicProfile(user: AppUser): PublicUserProfile {
    return {
      adRewardCount: user.adRewardCount,
      createdAt: user.createdAt,
      inviteCode: user.inviteCode,
      inviteRewardCount: user.inviteRewardCount,
      name: user.name,
      phone: user.phone,
      publicId: user.publicId,
      remainingCredits: user.remainingCredits,
      updatedAt: user.updatedAt,
      usedCredits: user.usedCredits,
    };
  }
}
