import { relations } from 'drizzle-orm';
import {
  AnyPgColumn,
  integer,
  jsonb,
  pgEnum,
  pgTable,
  serial,
  text,
  timestamp,
  uuid,
} from 'drizzle-orm/pg-core';

export const creditTransactionTypeEnum = pgEnum('credit_transaction_type', [
  'INITIAL_GRANT',
  'ANALYZE_CONSUME',
  'AD_REWARD',
  'INVITE_REWARD',
  'INVITEE_REWARD',
]);

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  publicId: text('public_id').notNull().unique(),
  openid: text('openid').notNull().unique(),
  unionid: text('unionid').unique(),
  sessionKey: text('session_key'),
  name: text('name'),
  phone: text('phone'),
  remainingCredits: integer('remaining_credits').notNull().default(3),
  usedCredits: integer('used_credits').notNull().default(0),
  inviteCode: text('invite_code').notNull().unique(),
  invitedByUserId: integer('invited_by_user_id').references(
    (): AnyPgColumn => users.id,
    { onDelete: 'set null' },
  ),
  adRewardCount: integer('ad_reward_count').notNull().default(0),
  inviteRewardCount: integer('invite_reward_count').notNull().default(0),
  createdAt: timestamp('created_at', { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true })
    .notNull()
    .defaultNow(),
});

export const creditTransactions = pgTable('credit_transactions', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: integer('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  type: creditTransactionTypeEnum('type').notNull(),
  delta: integer('delta').notNull(),
  balanceAfter: integer('balance_after').notNull(),
  metadata: jsonb('metadata').notNull().default({}),
  createdAt: timestamp('created_at', { withTimezone: true })
    .notNull()
    .defaultNow(),
});

export const usersRelations = relations(users, ({ one, many }) => ({
  invitedBy: one(users, {
    fields: [users.invitedByUserId],
    references: [users.id],
  }),
  invitees: many(users),
  creditTransactions: many(creditTransactions),
}));

export const creditTransactionsRelations = relations(
  creditTransactions,
  ({ one }) => ({
    user: one(users, {
      fields: [creditTransactions.userId],
      references: [users.id],
    }),
  }),
);
