import { relations } from 'drizzle-orm';
import {
  integer,
  jsonb,
  pgEnum,
  pgTable,
  real,
  text,
  timestamp,
  uuid,
} from 'drizzle-orm/pg-core';
import { users } from './users.schema';

export const analysisStatusEnum = pgEnum('analysis_status', [
  'PENDING',
  'DETECTING',
  'SCORING',
  'DONE',
  'FAILED',
]);

export const analysisTasks = pgTable('analysis_tasks', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: integer('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  sourceImagePath: text('source_image_path'),
  sourceImageUrl: text('source_image_url').notNull(),
  annotatedImageUrl: text('annotated_image_url'),
  detectedCount: integer('detected_count').notNull().default(0),
  detectedLabels: jsonb('detected_labels').notNull().default([]),
  status: analysisStatusEnum('status').notNull().default('PENDING'),
  errorMessage: text('error_message'),
  overallSummary: text('overall_summary'),
  recommendedLabel: text('recommended_label'),
  rawResult: jsonb('raw_result'),
  createdAt: timestamp('created_at', { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true })
    .notNull()
    .defaultNow(),
});

export const analysisTaskItems = pgTable('analysis_task_items', {
  id: uuid('id').defaultRandom().primaryKey(),
  taskId: uuid('task_id')
    .notNull()
    .references(() => analysisTasks.id, { onDelete: 'cascade' }),
  label: text('label').notNull(),
  bbox: jsonb('bbox').notNull(),
  confidence: real('confidence').notNull(),
  score: integer('score'),
  summary: text('summary'),
  reasons: jsonb('reasons'),
  risks: jsonb('risks'),
  buyPriority: integer('buy_priority'),
  createdAt: timestamp('created_at', { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true })
    .notNull()
    .defaultNow(),
});

export const analysisTaskRelations = relations(analysisTasks, ({ many }) => ({
  items: many(analysisTaskItems),
}));
