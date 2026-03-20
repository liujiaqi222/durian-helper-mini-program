import { relations } from 'drizzle-orm';
import {
  integer,
  jsonb,
  pgEnum,
  pgTable,
  text,
  timestamp,
  uuid,
} from 'drizzle-orm/pg-core';

export const analysisStatusEnum = pgEnum('analysis_status', [
  'PENDING',
  'DETECTING',
  'SCORING',
  'DONE',
  'FAILED',
]);

export const analysisTasks = pgTable('analysis_tasks', {
  id: uuid('id').defaultRandom().primaryKey(),
  sourceImagePath: text('source_image_path'),
  sourceImageUrl: text('source_image_url').notNull(),
  annotatedImageUrl: text('annotated_image_url'),
  status: analysisStatusEnum('status').notNull().default('PENDING'),
  errorMessage: text('error_message'),
  aiSummary: text('ai_summary'),
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
  score: integer('score'),
  summary: text('summary'),
  reasons: jsonb('reasons'),
  risks: jsonb('risks'),
  buyPriority: integer('buy_priority'),
  cropImageUrl: text('crop_image_url'),
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
