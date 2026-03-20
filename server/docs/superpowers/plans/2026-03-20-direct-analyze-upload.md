# Direct Analyze Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-step client upload flow with a single multipart analysis request that uploads the image and starts analysis immediately.

**Architecture:** The `POST /api/v1/durians/analyze` endpoint becomes the only upload entrypoint and accepts a `file` multipart field. The server stores the uploaded image, creates the task, and runs the existing CV pipeline. The client calls only this endpoint and keeps local preview state for the result page.

**Tech Stack:** NestJS, Multer, Taro, React, Jest

---

### Task 1: Lock the new server contract with tests

**Files:**
- Modify: `server/test/durians.e2e-spec.ts`
- Modify: `server/src/modules/durians/durians.service.spec.ts`

- [ ] Write failing e2e coverage for multipart `/api/v1/durians/analyze`.
- [ ] Remove JSON task creation expectations.
- [ ] Run tests to verify the contract fails before implementation.

### Task 2: Implement direct multipart analyze flow on the server

**Files:**
- Modify: `server/src/modules/durians/durians.controller.ts`
- Delete: `server/src/modules/uploads/uploads.controller.ts`
- Modify: `server/src/modules/uploads/uploads.module.ts`
- Modify: `server/src/modules/durians/dto/create-analysis-task.dto.ts`

- [ ] Accept uploaded files directly in `/durians/analyze`.
- [ ] Store the image via `UploadsService`.
- [ ] Pass stored image URL/path into `DuriansService`.

### Task 3: Update the client to call the single endpoint

**Files:**
- Modify: `client/src/services/api.ts`
- Modify: `client/src/pages/index/index.tsx`
- Modify: `client/src/store/analysis.ts`
- Modify: `client/src/types/analysis.ts`
- Modify: `client/src/pages/result/index.tsx`

- [ ] Remove the separate upload API call and related types/state.
- [ ] Upload the image file directly when the user taps “开始分析”.
- [ ] Keep local preview behavior and task polling intact.

### Task 4: Verify builds and regressions

**Files:**
- Modify only if verification exposes issues.

- [ ] Run server unit tests and e2e tests.
- [ ] Run `npm run build` in `server`.
- [ ] Run `npm run build:weapp` in `client`.
