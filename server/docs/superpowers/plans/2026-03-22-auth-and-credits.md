# Auth And Credits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build mini-program login, JWT auth, user profile, and a credit-based usage system that gives each new user 3 free analysis chances and supports ad/invite rewards.

**Architecture:** Add dedicated `auth` and `users` modules in NestJS, persist user identity and credit transactions in Postgres via Drizzle, and require JWT on analysis creation so one credit is consumed atomically per request. Update the Taro client to auto-login with `wx.login`, persist the JWT locally, show the current remaining chances, and call reward endpoints for ad/invite growth later.

**Tech Stack:** NestJS, Drizzle ORM, PostgreSQL, Taro React, Zustand, Jest, Vitest

---

### Task 1: Data Model And Server Tests

**Files:**
- Create: `server/src/database/drizzle/schema/users.schema.ts`
- Modify: `server/src/database/drizzle/schema/index.ts`
- Create: `server/test/auth-and-credits.e2e-spec.ts`
- Modify: `server/package.json`

- [ ] **Step 1: Write the failing test**
  Add e2e coverage for `POST /auth/login`, `GET /users/me`, credit reward endpoints, and authenticated `POST /durians/analyze`.

- [ ] **Step 2: Run test to verify it fails**
  Run: `cd server && npm test -- --runInBand --testPathPatterns auth-and-credits.e2e-spec.ts`
  Expected: FAIL because auth routes, auth dependencies, and user persistence do not exist yet.

- [ ] **Step 3: Write minimal implementation**
  Add `users`, `credit_transactions`, and optional reward-event fields so the service can create users, track balance, and prevent unlimited free usage.

- [ ] **Step 4: Run test to verify it passes**
  Run: `cd server && npm test -- --runInBand --testPathPatterns auth-and-credits.e2e-spec.ts`
  Expected: PASS

### Task 2: Auth, Users, Credits, And Protected Analyze

**Files:**
- Create: `server/src/modules/auth/*`
- Create: `server/src/modules/users/*`
- Modify: `server/src/modules/durians/durians.controller.ts`
- Modify: `server/src/modules/durians/durians.service.ts`
- Modify: `server/src/modules/durians/durians.types.ts`
- Modify: `server/src/app.module.ts`
- Modify: `server/src/config/index.ts`
- Modify: `server/src/config/env.ts`
- Modify: `server/drizzle/*.sql`

- [ ] **Step 1: Write the failing test**
  Extend tests to assert missing/invalid JWT is rejected and valid JWT causes a single credit deduction per analyze request.

- [ ] **Step 2: Run test to verify it fails**
  Run: `cd server && npm test -- --runInBand --testPathPatterns auth-and-credits.e2e-spec.ts`
  Expected: FAIL because protected analyze and credit deduction are not implemented.

- [ ] **Step 3: Write minimal implementation**
  Implement WeChat login abstraction, JWT signing/verifying, `GET /users/me`, `POST /users/me/rewards/ad`, `POST /users/me/rewards/invite`, repository methods, and atomic credit consumption before task creation.

- [ ] **Step 4: Run test to verify it passes**
  Run: `cd server && npm test -- --runInBand --testPathPatterns auth-and-credits.e2e-spec.ts`
  Expected: PASS

### Task 3: Client Login And Credit UX

**Files:**
- Create: `client/src/store/user.ts`
- Create: `client/src/types/user.ts`
- Modify: `client/src/services/api.ts`
- Modify: `client/src/app.ts`
- Modify: `client/src/pages/index/index.tsx`
- Modify: `client/src/pages/result/index.tsx`

- [ ] **Step 1: Write the failing test**
  Add or adapt unit tests around auth storage helpers or API header behavior.

- [ ] **Step 2: Run test to verify it fails**
  Run: `cd client && npm run test:unit`
  Expected: FAIL because auth helpers and credit-aware request flow do not exist.

- [ ] **Step 3: Write minimal implementation**
  Auto-login on launch, persist token and profile, attach `Authorization` headers, show remaining analysis chances, and expose reward actions for ad/invite increments.

- [ ] **Step 4: Run test to verify it passes**
  Run: `cd client && npm run test:unit`
  Expected: PASS

### Task 4: Verification

**Files:**
- Modify: `server/README.md`
- Modify: `client/Readme.md`

- [ ] **Step 1: Run targeted verification**
  Run: `cd server && npm test -- --runInBand --testPathPatterns 'app.e2e-spec.ts|durians.e2e-spec.ts|auth-and-credits.e2e-spec.ts'`

- [ ] **Step 2: Run client verification**
  Run: `cd client && npm run test:unit`

- [ ] **Step 3: Run build verification**
  Run: `cd server && npm run build`
  Run: `cd client && npm run build:weapp`

- [ ] **Step 4: Document required env vars**
  Note `WECHAT_APP_ID`, `WECHAT_APP_SECRET`, and `JWT_SECRET`, plus how reward endpoints currently work.
