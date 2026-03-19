CREATE TYPE "public"."analysis_status" AS ENUM('PENDING', 'DETECTING', 'SCORING', 'DONE', 'FAILED');--> statement-breakpoint
CREATE TABLE "analysis_task_items" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"task_id" uuid NOT NULL,
	"label" text NOT NULL,
	"score" integer,
	"summary" text,
	"reasons" jsonb,
	"risks" jsonb,
	"buy_priority" integer,
	"crop_image_url" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "analysis_tasks" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"source_image_url" text NOT NULL,
	"annotated_image_url" text,
	"status" "analysis_status" DEFAULT 'PENDING' NOT NULL,
	"error_message" text,
	"ai_summary" text,
	"recommended_label" text,
	"raw_result" jsonb,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "analysis_task_items" ADD CONSTRAINT "analysis_task_items_task_id_analysis_tasks_id_fk" FOREIGN KEY ("task_id") REFERENCES "public"."analysis_tasks"("id") ON DELETE cascade ON UPDATE no action;