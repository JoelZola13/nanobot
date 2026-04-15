# Street Voices Academy LMS Upgrade Notes

## Purpose
Street Voices Academy is moving from a branded learning area with partial demos into a focused, production-ready learning management system. This document is the Academy branch source of truth for the foundational LMS release.

## Product Goal
Build the ultimate learning management system for Street Voices Academy while staying strictly inside the Academy application and Academy pages. The work should improve navigation clarity, make Academy routes real and deep-linkable, connect learning content to live instruction, and give instructors a usable workspace for running courses.

## Scope Guardrails
- Work only in the Academy surface.
- Academy frontend work should stay primarily under `client/src/components/streetbot/academy/**`.
- Academy routing work should stay in `client/src/routes/index.tsx`.
- Backend changes should be limited to Academy API support such as `/api/academy/**`, `nanobot/api_server.py`, and Academy-specific tooling needed to support Academy flows.
- Do not redesign jobs, groups, directory, news, gallery, or global chat as part of this milestone.
- Do not remove the global Groups feature from the wider product. Only remove Academy-local links that incorrectly route learners there.

## Foundational LMS Release
This milestone focuses on a complete Academy baseline:
- Academy nav cleanup
- Academy landing cleanup
- real course detail pages
- real learning path detail pages
- real live-session detail pages
- cohorts embedded inside course detail
- working learner and instructor live-session workflows
- instructor workspace with grading, schedule, courses, examples, and AI course generation
- Academy API normalization under `/api/academy/**`

## Canonical Route Structure
`/academy` is the canonical route family.

`/learning/*` remains supported as a compatibility alias that resolves to the same Academy experience.

### Required Academy Routes
- `/academy`
- `/academy/courses`
- `/academy/courses/:courseId`
- `/academy/paths`
- `/academy/paths/:slug`
- `/academy/live-sessions`
- `/academy/live-sessions/:sessionId`
- `/academy/instructor`
- `/academy/instructor/grading`
- `/academy/instructor/schedule`
- `/academy/instructor/courses`
- `/academy/instructor/examples`
- `/academy/instructor/generate`

## Navigation and Information Architecture
### Sidebar and layout
- Fix the sidebar and layout so the Academy logo remains visible on desktop and mobile.
- Remove Academy links that route to `/groups`.
- Top-level Academy nav should only expose working destinations.

### Sidebar order
The Academy navigation should be:
1. Learning Paths
2. Courses
3. Live Sessions
4. Instructor

### Remove as standalone top-level nav items
These should not remain dead-end Academy sidebar destinations:
- assignments
- peer review
- attendance
- progress
- certificates
- AI tutor
- discussions

These belong as sections within course detail, live-session detail, or the instructor workspace.

## Academy Landing Page
### Required changes
- Remove the top-right `Get Started` CTA.
- Change the hero CTA text to exactly `Start your first course free`.
- Remove the public-facing marketing stats block.
- Keep learner-specific progress widgets only if they appear inside authenticated dashboard content and not as public landing statistics.

### Academy identity
- The landing experience should feel Academy-specific rather than a generic promo page.
- Academy links should drive users deeper into learning flows rather than to unrelated Street Voices sections.

## Courses
### List page
- Keep `/academy/courses` as the course catalog.
- Course cards must link to real course detail pages.
- Remove `Ask` and `Ask About Course` actions from course cards.

### Detail page requirements
`/academy/courses/:courseId` is the primary LMS hub.

Each course detail page should include:
- course header and metadata
- enrollment state and progress
- curriculum with ordered modules and lessons
- cohorts for the selected course
- live sessions for the selected course
- assignments
- materials
- reviews
- certificate area
- discussions
- offline/download controls

### Cohorts
- Cohorts belong inside course detail.
- Do not keep cohorts as a standalone Academy destination for this milestone.
- Cohort enrollment and deadlines should be contextual to the course.

## Learning Paths
### List page
- `/academy/paths` remains the list of learning paths.
- Cards must route to real path detail pages.

### Detail page requirements
Each path detail page should show:
- path overview
- ordered course sequence
- milestone structure
- total time
- learner progress
- start or continue behavior

### Start and continue behavior
- `Start path` and `Continue path` should send learners to the first incomplete course or lesson.

## Live Sessions
### Frontend model
- Standardize Academy live-session UX on `client/src/components/streetbot/academy/api/live-sessions.ts`.
- The older split attendance model should not be the primary live-session source for Academy pages.
- Attendance logic may remain as support, but the Academy session surface must be driven by the live-session adapter.

### Required user flows
Learners must be able to:
- register
- join
- leave
- ask questions
- answer polls
- submit feedback

Instructors must be able to:
- create sessions
- start sessions
- end sessions

### Required pages
- `/academy/live-sessions` should show real upcoming, live, and past sessions.
- `/academy/live-sessions/:sessionId` should render `LiveSessionViewer`.

## Instructor Workspace
The instructor workspace remains centered at `/academy/instructor`, but it must become a route-backed workspace rather than a two-tab demo.

### Required instructor destinations
- grading
- schedule
- courses
- examples
- AI course generation

### Schedule requirements
The instructor schedule should provide a calendar or schedule-oriented view of:
- live sessions
- workshops
- cohort deadlines
- assignment deadlines
- related Academy events

### Instructor courses requirements
- The instructor courses destination should show only instructor-owned courses.

### Examples requirements
- Provide example students and submission samples.
- Use real data when available.
- When sample data is needed, label it clearly as sample or demo data.

## Data and API Normalization
### Canonical frontend data surface
- `/api/academy/courses`
- `/api/academy/courses/:courseId`
- `/api/academy/courses/:courseId/modules`
- `/api/academy/modules/:moduleId/lessons`
- `/api/academy/courses/:courseId/cohorts`
- `/api/academy/courses/:courseId/live-sessions`
- `/api/academy/live-sessions/**`
- `/api/academy/cohorts/**`

### Backend expectations
Extend `nanobot/api_server.py` so Academy routes return the shapes Academy pages expect instead of falling back to disconnected sample behavior.

### Live-session endpoints to support
- `GET /api/academy/live-sessions`
- `POST /api/academy/live-sessions`
- `GET /api/academy/live-sessions/:sessionId`
- `PATCH /api/academy/live-sessions/:sessionId`
- `POST /api/academy/live-sessions/:sessionId/start`
- `POST /api/academy/live-sessions/:sessionId/end`
- `POST /api/academy/live-sessions/:sessionId/register`
- `POST /api/academy/live-sessions/:sessionId/join`
- `POST /api/academy/live-sessions/:sessionId/leave`
- poll endpoints
- Q&A endpoints
- feedback endpoints

### Cohort endpoints to support
- list cohorts
- get cohorts for a course
- enroll in cohort
- list cohort enrollments
- list cohort deadlines
- list user cohorts and deadlines
- create and list cohort announcements
- cohort analytics

### Supporting course-adjacent APIs
The foundational release should also support the Academy components that appear on course detail:
- reviews
- certificate checks and auto-issue
- course materials
- video progress
- discussion/forum support where already represented by Academy UI

## Identity and User State
- Replace hardcoded Academy learner IDs such as `user-123` and `demo-user` in primary learner flows.
- Use auth-derived user identity when available.
- Persist Academy user identity locally only as a compatibility fallback.
- Demo identifiers are acceptable only in clearly labeled sample/example views.

## UI Contract Decisions
- Top-level Academy nav exposes only destinations with routed, working pages.
- Assignments, discussions, reviews, certificates, and materials are embedded sections, not first-phase top-level nav items.
- Course detail becomes the main learner workspace.
- Live-session detail becomes the main synchronous learning workspace.
- Instructor sub-routes become the main teaching workspace.

## Phased Delivery
### Phase 1: Academy shell and route integrity
- fix logo and sidebar behavior
- clean Academy nav
- remove dead-end links
- update landing CTA and stats block
- add canonical route coverage with compatibility aliases

### Phase 2: Course and path depth
- real course detail pages
- real path detail pages
- course curriculum and learner progress
- cohorts inside course detail
- course-adjacent sections for assignments, materials, reviews, certificates, and discussions

### Phase 3: Live learning
- live sessions index and detail pages
- learner register, join, leave, questions, polls, feedback
- instructor start and end controls

### Phase 4: Instructor operations
- grading workspace
- schedule view
- instructor-owned courses
- examples area
- AI course generation

### Phase 5: API normalization and cleanup
- fill Academy API gaps in `api_server.py`
- normalize cohorts and live sessions under `/api/academy/**`
- remove hardcoded IDs from primary Academy flows

## Backend Checklist
- Academy proxy responds for the new live-session endpoints.
- Academy proxy responds for the new cohort endpoints.
- Academy proxy can resolve course detail, nested modules, and nested lessons cleanly.
- Reviews endpoints exist for course detail.
- Certificate endpoints exist for course detail.
- Course materials endpoints are Academy-scoped.
- No primary Academy page depends on the older split `sessions` and `engagement` data model.
- No production learner flow depends on `user-123` or `demo-user`.

## Definition of Done
### Landing
- Academy logo remains visible across desktop and mobile.
- No Academy Community or Groups link remains in the Academy shell.
- No top-right `Get Started` button remains.
- Hero CTA reads exactly `Start your first course free`.
- Public stats block is removed.
- Nav order is Learning Paths before Courses.

### Routing
- `/academy/courses/:courseId` deep-links and refreshes correctly.
- `/academy/paths/:slug` deep-links and refreshes correctly.
- `/academy/live-sessions/:sessionId` deep-links and refreshes correctly.
- Invalid course, path, and session IDs render Academy-scoped not-found states.
- `/learning/*` aliases resolve to the same Academy destinations.

### Course hub
- Course cards open real detail pages.
- Cohorts render inside course detail and enroll against the selected course.
- Course sections for assignments, materials, reviews, discussions, and certificates render without bouncing back to list pages.

### Live sessions
- Instructor can start and end eligible sessions.
- Learner can register, join, leave, ask a question, answer a poll, and submit feedback.
- Session state updates without a full page reload.

### Instructor workspace
- Schedule shows Academy events and links back to related course, cohort, or session detail.
- Instructor courses list filters to the active instructor.
- Examples tab shows real data when available or clearly labeled sample data when not.

## Notes for This Branch
- This markdown file is the Academy reference spec for the `academy` branch.
- This milestone is intentionally scoped to the foundational LMS release, not every future Academy ambition in one pass.
- Any new Academy page or API addition should be evaluated against this document before expanding scope.
