# Frontend Annotation Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove backend-generated annotated images and render durian labels and highlight state entirely on the client using source-image coordinates.

**Architecture:** The Python CV service will keep returning only structured detections plus crop images for AI scoring. The Nest server will stop persisting or exposing annotated-image URLs. The client result page will render the original uploaded image and draw letter badges plus bounding boxes from `bbox`, with the `recommendedLabel` highlighted by default after scoring completes.

**Tech Stack:** FastAPI, Pydantic, NestJS, Jest, Taro React, TypeScript

---

### Task 1: Remove annotated-image generation from `cv-service`

**Files:**
- Modify: `cv-service/app/services/detector.py`
- Modify: `cv-service/app/schemas.py`
- Test: `cv-service/tests/test_detect.py`
- Test: `cv-service/tests/test_sorting.py`

- [ ] **Step 1: Write the failing tests**
Add assertions that `/detect-and-annotate` no longer returns `annotated_image_base64`, while still returning `count`, `items`, labels, bbox, and crop images.

- [ ] **Step 2: Run tests to verify they fail**
Run: `cd cv-service && pytest tests/test_detect.py tests/test_sorting.py -q`
Expected: failures referencing the old annotated-image field.

- [ ] **Step 3: Write minimal implementation**
Delete annotated-image generation and remove `annotated_image_base64` from the response schema and builder path.

- [ ] **Step 4: Run tests to verify they pass**
Run: `cd cv-service && pytest tests/test_detect.py tests/test_sorting.py -q`
Expected: PASS.

### Task 2: Remove annotated-image persistence and exposure from `server`

**Files:**
- Modify: `server/src/modules/durians/cv.service.ts`
- Modify: `server/src/modules/durians/durians.service.ts`
- Modify: `server/src/modules/durians/durians.types.ts`
- Modify: `server/src/modules/durians/cv.service.spec.ts`
- Modify: `server/src/modules/durians/durians.service.spec.ts`
- Modify: `server/test/durians.e2e-spec.ts`

- [ ] **Step 1: Write the failing tests**
Update server tests to expect no annotated-image handling from the CV payload and no annotated-image URL in task/result responses.

- [ ] **Step 2: Run tests to verify they fail**
Run: `cd server && npm test -- --runInBand src/modules/durians/cv.service.spec.ts src/modules/durians/durians.service.spec.ts test/durians.e2e-spec.ts`
Expected: failures where tests still expect annotated-image fields.

- [ ] **Step 3: Write minimal implementation**
Remove annotated-image parsing, storage, raw-result hydration, and response fields while preserving source-image URLs and detection metadata.

- [ ] **Step 4: Run tests to verify they pass**
Run: `cd server && npm test -- --runInBand src/modules/durians/cv.service.spec.ts src/modules/durians/durians.service.spec.ts test/durians.e2e-spec.ts`
Expected: PASS.

### Task 3: Render detection overlays on the client

**Files:**
- Modify: `client/src/types/analysis.ts`
- Modify: `client/src/utils/analysis.ts`
- Modify: `client/src/utils/analysis.test.ts`
- Modify: `client/src/pages/result/index.tsx`

- [ ] **Step 1: Write the failing tests**
Add utility tests for preview resolution using only the source image and for default selection using `recommendedLabel`.

- [ ] **Step 2: Run tests to verify they fail**
Run: `cd client && npm test -- --runInBand src/utils/analysis.test.ts`
Expected: failures because preview logic still prefers annotated images or because selection helpers are missing.

- [ ] **Step 3: Write minimal implementation**
Use the source image as the only preview, measure rendered image dimensions, scale bbox coordinates into overlay positions, render centered letter badges and boxes, and highlight the recommended label by default.

- [ ] **Step 4: Run tests to verify they pass**
Run: `cd client && npm test -- --runInBand src/utils/analysis.test.ts`
Expected: PASS.

### Task 4: Verify the integrated behavior

**Files:**
- Modify: `server/docs/analyze-接口现状梳理.md`

- [ ] **Step 1: Update docs**
Replace references to backend annotated-image generation with source-image overlay rendering on the client.

- [ ] **Step 2: Run focused verification**
Run:
`cd cv-service && pytest tests/test_detect.py tests/test_sorting.py -q`
`cd server && npm test -- --runInBand src/modules/durians/cv.service.spec.ts src/modules/durians/durians.service.spec.ts test/durians.e2e-spec.ts`
`cd client && npm test -- --runInBand src/utils/analysis.test.ts`
Expected: all targeted suites pass.
