import {
  BadRequestException,
  ConflictException,
  Inject,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { and, eq, isNull, or } from 'drizzle-orm';
import { DRIZZLE, type DrizzleDb } from '../../database/drizzle/drizzle.module';
import {
  creditTransactions,
  users,
} from '../../database/drizzle/schema';
import {
  AD_REWARD_CREDITS,
  INITIAL_FREE_CREDITS,
} from './users.constants';
import type {
  AppUser,
  CreateWechatUserInput,
  CreditTransactionType,
  UpdateWechatSessionInput,
  UsersRepository,
} from './users.types';

type TransactionClient = Pick<DrizzleDb, 'insert' | 'select' | 'update'>;

@Injectable()
export class DrizzleUsersRepository implements UsersRepository {
  constructor(@Inject(DRIZZLE) private readonly db: DrizzleDb) {}

  async createWechatUser(input: CreateWechatUserInput): Promise<AppUser> {
    return this.db.transaction(async (tx) => {
      const [user] = await tx
        .insert(users)
        .values({
          inviteCode: this.buildInviteCode(),
          openid: input.openid,
          publicId: this.buildPublicId(),
          sessionKey: input.sessionKey ?? null,
          unionid: input.unionid ?? null,
        })
        .returning();

      await tx.insert(creditTransactions).values({
        balanceAfter: INITIAL_FREE_CREDITS,
        delta: INITIAL_FREE_CREDITS,
        metadata: { source: 'mini-program-first-login' },
        type: 'INITIAL_GRANT',
        userId: user.id,
      });

      return this.mapUser(user);
    });
  }

  async findByInviteCode(inviteCode: string): Promise<AppUser | null> {
    const [user] = await this.db
      .select()
      .from(users)
      .where(eq(users.inviteCode, inviteCode))
      .limit(1);
    return user ? this.mapUser(user) : null;
  }

  async findByOpenidOrUnionid(
    openid: string,
    unionid?: string | null,
  ): Promise<AppUser | null> {
    const where = unionid
      ? or(eq(users.openid, openid), eq(users.unionid, unionid))
      : eq(users.openid, openid);

    const [user] = await this.db.select().from(users).where(where).limit(1);
    return user ? this.mapUser(user) : null;
  }

  async findByPublicId(publicId: string): Promise<AppUser | null> {
    const [user] = await this.db
      .select()
      .from(users)
      .where(eq(users.publicId, publicId))
      .limit(1);
    return user ? this.mapUser(user) : null;
  }

  async findByUserId(userId: number): Promise<AppUser | null> {
    const [user] = await this.db
      .select()
      .from(users)
      .where(eq(users.id, userId))
      .limit(1);
    return user ? this.mapUser(user) : null;
  }

  async grantAdReward(userId: number): Promise<AppUser> {
    return this.db.transaction(async (tx) => {
      const user = await this.requireUser(tx, userId);
      const nextRemainingCredits = user.remainingCredits + AD_REWARD_CREDITS;

      const [updated] = await tx
        .update(users)
        .set({
          adRewardCount: user.adRewardCount + 1,
          remainingCredits: nextRemainingCredits,
          updatedAt: new Date(),
        })
        .where(eq(users.id, userId))
        .returning();

      await this.insertCreditTransaction(tx, {
        balanceAfter: nextRemainingCredits,
        delta: AD_REWARD_CREDITS,
        metadata: { source: 'ad' },
        type: 'AD_REWARD',
        userId,
      });

      return this.mapUser(updated);
    });
  }

  async bindInviter(input: {
    userId: number;
    inviterCode: string;
    inviteeRewardCredits: number;
    inviterRewardCredits: number;
  }): Promise<AppUser> {
    return this.db.transaction(async (tx) => {
      const invitee = await this.requireUser(tx, input.userId);
      if (invitee.invitedByUserId) {
        throw new ConflictException('invite reward already claimed');
      }

      const [inviter] = await tx
        .select()
        .from(users)
        .where(eq(users.inviteCode, input.inviterCode))
        .limit(1);

      if (!inviter) {
        throw new NotFoundException('inviter not found');
      }
      if (inviter.id === invitee.id) {
        throw new BadRequestException('cannot use your own invite code');
      }

      const now = new Date();
      const inviteeBalance = invitee.remainingCredits + input.inviteeRewardCredits;
      const inviterBalance = inviter.remainingCredits + input.inviterRewardCredits;

      const [updatedInvitee] = await tx
        .update(users)
        .set({
          invitedByUserId: inviter.id,
          remainingCredits: inviteeBalance,
          updatedAt: now,
        })
        .where(and(eq(users.id, invitee.id), isNull(users.invitedByUserId)))
        .returning();

      if (!updatedInvitee) {
        throw new ConflictException('invite reward already claimed');
      }

      await tx
        .update(users)
        .set({
          inviteRewardCount: inviter.inviteRewardCount + 1,
          remainingCredits: inviterBalance,
          updatedAt: now,
        })
        .where(eq(users.id, inviter.id));

      await this.insertCreditTransaction(tx, {
        balanceAfter: inviteeBalance,
        delta: input.inviteeRewardCredits,
        metadata: { inviterCode: input.inviterCode, role: 'invitee' },
        type: 'INVITEE_REWARD',
        userId: invitee.id,
      });
      await this.insertCreditTransaction(tx, {
        balanceAfter: inviterBalance,
        delta: input.inviterRewardCredits,
        metadata: { inviteePublicId: invitee.publicId, role: 'inviter' },
        type: 'INVITE_REWARD',
        userId: inviter.id,
      });

      return this.mapUser(updatedInvitee);
    });
  }

  async consumeCredit(input: {
    userId: number;
    reason: string;
  }): Promise<AppUser> {
    return this.db.transaction(async (tx) => {
      const user = await this.requireUser(tx, input.userId);
      if (user.remainingCredits <= 0) {
        throw new BadRequestException('remaining credits are insufficient');
      }

      const nextRemainingCredits = user.remainingCredits - 1;
      const nextUsedCredits = user.usedCredits + 1;
      const [updated] = await tx
        .update(users)
        .set({
          remainingCredits: nextRemainingCredits,
          updatedAt: new Date(),
          usedCredits: nextUsedCredits,
        })
        .where(and(eq(users.id, input.userId)))
        .returning();

      await this.insertCreditTransaction(tx, {
        balanceAfter: nextRemainingCredits,
        delta: -1,
        metadata: { reason: input.reason },
        type: 'ANALYZE_CONSUME',
        userId: input.userId,
      });

      return this.mapUser(updated);
    });
  }

  async updateWechatSession(input: UpdateWechatSessionInput): Promise<AppUser> {
    const [user] = await this.db
      .update(users)
      .set({
        sessionKey: input.sessionKey ?? undefined,
        unionid: input.unionid ?? undefined,
        updatedAt: new Date(),
      })
      .where(eq(users.id, input.userId))
      .returning();

    if (!user) {
      throw new NotFoundException('user not found');
    }
    return this.mapUser(user);
  }

  private async insertCreditTransaction(
    tx: TransactionClient,
    value: {
      userId: number;
      type: CreditTransactionType;
      delta: number;
      balanceAfter: number;
      metadata: Record<string, unknown>;
    },
  ): Promise<void> {
    await tx.insert(creditTransactions).values(value);
  }

  private async requireUser(
    tx: TransactionClient,
    userId: number,
  ) {
    const [user] = await tx.select().from(users).where(eq(users.id, userId)).limit(1);
    if (!user) {
      throw new NotFoundException('user not found');
    }
    return user;
  }

  private buildInviteCode(): string {
    return `INV${randomSegment(6).toUpperCase()}`;
  }

  private buildPublicId(): string {
    return `usr_${randomSegment(12)}`;
  }

  private mapUser(user: typeof users.$inferSelect): AppUser {
    return {
      adRewardCount: user.adRewardCount,
      createdAt: user.createdAt,
      id: user.id,
      inviteCode: user.inviteCode,
      invitedByUserId: user.invitedByUserId,
      inviteRewardCount: user.inviteRewardCount,
      name: user.name,
      openid: user.openid,
      phone: user.phone,
      publicId: user.publicId,
      remainingCredits: user.remainingCredits,
      sessionKey: user.sessionKey,
      unionid: user.unionid,
      updatedAt: user.updatedAt,
      usedCredits: user.usedCredits,
    };
  }
}

function randomSegment(length: number): string {
  return Math.random().toString(36).slice(2, 2 + length).padEnd(length, '0');
}
