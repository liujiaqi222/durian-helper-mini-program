# Durian Backend Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap a runnable NestJS backend in `server/` with Postgres, Redis, Winston, Vercel AI SDK, and durian analysis task API scaffolding.

**Architecture:** Create a small global `CoreModule` for config, logging, exception handling, and response transforms. Add isolated `database`, `ai`, and `durians` modules, with Drizzle-backed task persistence and REST endpoints under `/api/v1`.

**Tech Stack:** NestJS, TypeScript, Drizzle ORM, PostgreSQL, Redis, ioredis, Winston, Vercel AI SDK, Jest.

---

### Task 1: Scaffold project

**Files:**
- Create: `server/package.json`, `server/src/**`, `server/test/**`, `server/tsconfig*.json`, `server/nest-cli.json`

- [ ] Initialize NestJS package structure and scripts.
- [ ] Install runtime and dev dependencies.
- [ ] Add base TypeScript, Jest, ESLint, and Prettier config.

### Task 2: Add shared runtime infrastructure

**Files:**
- Create: `server/src/config/index.ts`
- Create: `server/src/core/**`
- Create: `server/src/database/**`

- [ ] Add typed env config with required URLs and AI settings.
- [ ] Add Winston logger, global exception filter, and response transform interceptor.
- [ ] Add Postgres and Redis providers/modules.

### Task 3: Add durian task domain

**Files:**
- Create: `server/src/modules/durians/**`
- Create: `server/src/database/drizzle/schema/**`
- Create: `server/drizzle.config.ts`

- [ ] Write failing tests for task creation and retrieval behavior.
- [ ] Add Drizzle schema and repository/service/controller code.
- [ ] Generate initial migration files.

### Task 4: Add AI module and minimal integration surface

**Files:**
- Create: `server/src/modules/ai/**`

- [ ] Add Vercel AI SDK client for Doubao via Anthropic-compatible endpoint.
- [ ] Expose a service method for future structured scoring.
- [ ] Keep task APIs decoupled from execution pipeline for now.

### Task 5: Verify

**Files:**
- Modify: `server/.env.example`

- [ ] Run targeted tests first and confirm red-green cycle.
- [ ] Run build.
- [ ] Run full test suite if available.
