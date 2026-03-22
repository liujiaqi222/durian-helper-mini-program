import { randomUUID } from 'crypto';
import {
  BadRequestException,
  ConflictException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import {
  AD_REWARD_CREDITS,
  INITIAL_FREE_CREDITS,
} from './users.constants';
import type {
  AppUser,
  CreateWechatUserInput,
  UpdateWechatSessionInput,
  UsersRepository,
} from './users.types';

@Injectable()
export class InMemoryUsersRepository implements UsersRepository {
  private readonly usersById = new Map<number, AppUser>();
  private nextId = 1;

  async createWechatUser(input: CreateWechatUserInput): Promise<AppUser> {
    const now = new Date();
    const user: AppUser = {
      id: this.nextId++,
      publicId: `usr_${randomUUID().replace(/-/g, '').slice(0, 12)}`,
      openid: input.openid,
      unionid: input.unionid ?? null,
      sessionKey: input.sessionKey ?? null,
      name: null,
      phone: null,
      remainingCredits: INITIAL_FREE_CREDITS,
      usedCredits: 0,
      inviteCode: `INV${randomUUID().replace(/-/g, '').slice(0, 6).toUpperCase()}`,
      invitedByUserId: null,
      adRewardCount: 0,
      inviteRewardCount: 0,
      createdAt: now,
      updatedAt: now,
    };

    this.usersById.set(user.id, user);
    return this.cloneUser(user);
  }

  async findByInviteCode(inviteCode: string): Promise<AppUser | null> {
    for (const user of this.usersById.values()) {
      if (user.inviteCode === inviteCode) {
        return this.cloneUser(user);
      }
    }
    return null;
  }

  async findByOpenidOrUnionid(
    openid: string,
    unionid?: string | null,
  ): Promise<AppUser | null> {
    for (const user of this.usersById.values()) {
      if (user.openid === openid) {
        return this.cloneUser(user);
      }
      if (unionid && user.unionid && user.unionid === unionid) {
        return this.cloneUser(user);
      }
    }
    return null;
  }

  async findByPublicId(publicId: string): Promise<AppUser | null> {
    for (const user of this.usersById.values()) {
      if (user.publicId === publicId) {
        return this.cloneUser(user);
      }
    }
    return null;
  }

  async findByUserId(userId: number): Promise<AppUser | null> {
    const user = this.usersById.get(userId);
    return user ? this.cloneUser(user) : null;
  }

  async grantAdReward(userId: number): Promise<AppUser> {
    const user = this.requireUser(userId);
    const nextUser: AppUser = {
      ...user,
      adRewardCount: user.adRewardCount + 1,
      remainingCredits: user.remainingCredits + AD_REWARD_CREDITS,
      updatedAt: new Date(),
    };
    this.usersById.set(userId, nextUser);
    return this.cloneUser(nextUser);
  }

  async bindInviter(input: {
    userId: number;
    inviterCode: string;
    inviteeRewardCredits: number;
    inviterRewardCredits: number;
  }): Promise<AppUser> {
    const invitee = this.requireUser(input.userId);
    if (invitee.invitedByUserId) {
      throw new ConflictException('invite reward already claimed');
    }

    const inviter = await this.findByInviteCode(input.inviterCode);
    if (!inviter) {
      throw new NotFoundException('inviter not found');
    }
    if (inviter.id === invitee.id) {
      throw new BadRequestException('cannot use your own invite code');
    }

    const now = new Date();
    const nextInvitee: AppUser = {
      ...invitee,
      invitedByUserId: inviter.id,
      remainingCredits: invitee.remainingCredits + input.inviteeRewardCredits,
      updatedAt: now,
    };
    const nextInviter: AppUser = {
      ...inviter,
      inviteRewardCount: inviter.inviteRewardCount + 1,
      remainingCredits: inviter.remainingCredits + input.inviterRewardCredits,
      updatedAt: now,
    };

    this.usersById.set(nextInvitee.id, nextInvitee);
    this.usersById.set(nextInviter.id, nextInviter);
    return this.cloneUser(nextInvitee);
  }

  async consumeCredit(input: {
    userId: number;
    reason: string;
  }): Promise<AppUser> {
    const user = this.requireUser(input.userId);
    if (user.remainingCredits <= 0) {
      throw new BadRequestException('remaining credits are insufficient');
    }

    const nextUser: AppUser = {
      ...user,
      remainingCredits: user.remainingCredits - 1,
      usedCredits: user.usedCredits + 1,
      updatedAt: new Date(),
    };
    this.usersById.set(user.id, nextUser);
    return this.cloneUser(nextUser);
  }

  async updateWechatSession(input: UpdateWechatSessionInput): Promise<AppUser> {
    const user = this.requireUser(input.userId);
    const nextUser: AppUser = {
      ...user,
      sessionKey: input.sessionKey ?? user.sessionKey,
      unionid: input.unionid ?? user.unionid,
      updatedAt: new Date(),
    };
    this.usersById.set(user.id, nextUser);
    return this.cloneUser(nextUser);
  }

  private requireUser(userId: number): AppUser {
    const user = this.usersById.get(userId);
    if (!user) {
      throw new NotFoundException('user not found');
    }
    return user;
  }

  private cloneUser(user: AppUser): AppUser {
    return {
      ...user,
      createdAt: new Date(user.createdAt),
      updatedAt: new Date(user.updatedAt),
    };
  }
}
