CREATE TYPE "public"."credit_transaction_type" AS ENUM('INITIAL_GRANT', 'ANALYZE_CONSUME', 'AD_REWARD', 'INVITE_REWARD', 'INVITEE_REWARD');--> statement-breakpoint
CREATE TABLE "credit_transactions" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" integer NOT NULL,
	"type" "credit_transaction_type" NOT NULL,
	"delta" integer NOT NULL,
	"balance_after" integer NOT NULL,
	"metadata" jsonb DEFAULT '{}'::jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" serial PRIMARY KEY NOT NULL,
	"public_id" text NOT NULL,
	"openid" text NOT NULL,
	"unionid" text,
	"session_key" text,
	"name" text,
	"phone" text,
	"remaining_credits" integer DEFAULT 3 NOT NULL,
	"used_credits" integer DEFAULT 0 NOT NULL,
	"invite_code" text NOT NULL,
	"invited_by_user_id" integer,
	"ad_reward_count" integer DEFAULT 0 NOT NULL,
	"invite_reward_count" integer DEFAULT 0 NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "users_public_id_unique" UNIQUE("public_id"),
	CONSTRAINT "users_openid_unique" UNIQUE("openid"),
	CONSTRAINT "users_unionid_unique" UNIQUE("unionid"),
	CONSTRAINT "users_invite_code_unique" UNIQUE("invite_code")
);
--> statement-breakpoint
ALTER TABLE "analysis_tasks" ALTER COLUMN "detected_count" SET DEFAULT 0;--> statement-breakpoint
ALTER TABLE "analysis_tasks" ADD COLUMN "user_id" integer NOT NULL;--> statement-breakpoint
ALTER TABLE "credit_transactions" ADD CONSTRAINT "credit_transactions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "users" ADD CONSTRAINT "users_invited_by_user_id_users_id_fk" FOREIGN KEY ("invited_by_user_id") REFERENCES "public"."users"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "analysis_tasks" ADD CONSTRAINT "analysis_tasks_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;