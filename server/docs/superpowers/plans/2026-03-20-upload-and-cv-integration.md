# Upload And CV Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add server-side image upload and static serving, persist both image URL and local path on analysis tasks, and trigger CV detection using local files when available.

**Architecture:** Keep a separate upload endpoint that stores files in `server/uploads/` and returns both public URL and local path. Extend the durian task service with a CV client that prefers multipart file upload to `cv-service`, falls back to URL mode, and stores the raw CV response on the task for later AI/result stages.

**Tech Stack:** NestJS, Multer, Express static assets, FastAPI, pytest, Jest

---

### Task 1: Lock the server contract with tests

**Files:**
- Modify: `server/src/modules/durians/durians.service.spec.ts`
- Modify: `server/test/durians.e2e-spec.ts`

- [ ] Add failing tests for analysis task creation with uploaded image metadata.
- [ ] Add a failing e2e test for `POST /api/v1/uploads/images`.
- [ ] Add a failing e2e assertion that `/api/v1/durians/analyze` still accepts direct `imageUrl`.

### Task 2: Implement upload storage and static exposure

**Files:**
- Modify: `server/src/main.ts`
- Create: `server/src/modules/uploads/uploads.controller.ts`
- Create: `server/src/modules/uploads/uploads.service.ts`
- Create: `server/src/modules/uploads/uploads.module.ts`
- Create: `server/src/modules/uploads/dto/upload-image-response.dto.ts`
- Modify: `server/src/app.module.ts`

- [ ] Add an upload endpoint that saves images to `server/uploads/`.
- [ ] Return both public `fileUrl` and absolute `localPath`.
- [ ] Expose `/uploads/*` as static assets.

### Task 3: Persist local paths and call CV service

**Files:**
- Modify: `server/src/config/index.ts`
- Modify: `server/src/modules/durians/durians.types.ts`
- Modify: `server/src/modules/durians/durians.memory-repository.ts`
- Modify: `server/src/modules/durians/durians.repository.ts`
- Modify: `server/src/modules/durians/durians.service.ts`
- Modify: `server/src/modules/durians/durians.controller.ts`
- Modify: `server/src/modules/durians/dto/create-analysis-task.dto.ts`
- Modify: `server/src/database/drizzle/schema/analysis-tasks.schema.ts`
- Create: `server/src/modules/durians/cv.service.ts`

- [ ] Extend task creation input to accept `imageUrl` and optional `imagePath`.
- [ ] Save `sourceImagePath` on tasks.
- [ ] Add a CV client that uploads a local file to `/detect-and-annotate`, or falls back to `image_url`.
- [ ] Update task status and raw result based on CV response.

### Task 4: Verify cv-service compatibility

**Files:**
- Modify: `cv-service/tests/test_detect.py` (only if coverage is needed for server call shape)
- Modify: `cv-service/README.md`

- [ ] Confirm `cv-service` continues to accept multipart uploads from Nest.
- [ ] Add or adjust tests only if the server integration needs an explicit contract.
- [ ] Document the new server-to-cv-service flow.
