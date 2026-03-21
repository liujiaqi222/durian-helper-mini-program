ALTER TABLE "analysis_tasks" ADD COLUMN "detected_count" integer DEFAULT 0 NOT NULL;
ALTER TABLE "analysis_tasks" ADD COLUMN "detected_labels" jsonb DEFAULT '[]'::jsonb NOT NULL;
ALTER TABLE "analysis_tasks" RENAME COLUMN "ai_summary" TO "overall_summary";
ALTER TABLE "analysis_task_items" ADD COLUMN "bbox" jsonb DEFAULT '{"x1":0,"y1":0,"x2":0,"y2":0}'::jsonb NOT NULL;
ALTER TABLE "analysis_task_items" ADD COLUMN "confidence" real DEFAULT 0 NOT NULL;
ALTER TABLE "analysis_task_items" DROP COLUMN "crop_image_url";
