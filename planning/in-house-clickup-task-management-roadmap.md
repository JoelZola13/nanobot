# In-House ClickUp Task Management Roadmap

Last reviewed: 2026-05-04

## Purpose

Build Nanobot into an in-house task management system with ClickUp-level depth while keeping our own agent-native workflows, grant operations, and Street Voices context at the center.

This file is the living product and engineering roadmap. Update it after every implementation cycle with shipped features, changed assumptions, and newly discovered gaps.

## North Star

Nanobot should become the single place where work is captured, structured, assigned, automated, discussed, measured, and completed.

The app must support a full hierarchy from broad operating areas down to fine-grained subtasks:

```text
Workspace: Street Voices / Nanobot
  Space: Operations
    Folder: Grant
      List: To do grants
        Task: Toronto Arts Council
          Subtask: Fill out the budget
```

Every layer must be meaningful. A user should be able to navigate, permission, filter, automate, report on, and attach context at the correct hierarchy level.

## Current Reference Baseline

ClickUp capabilities to treat as the comparison baseline:

- Hierarchy: Workspace, Spaces, Folders, Lists, Tasks, Subtasks, nested subtasks, plus hierarchy items like Docs, Dashboards, Forms, and Whiteboards.
- Tasks live in Lists, can have subtasks and nested subtasks, and can be added to more than one List with one home List.
- Folders are optional but useful for complex work. Folders contain Lists, not tasks directly.
- Views exist at Space, Folder, and List levels. Required task views include List, Board, Calendar, Team, Gantt, Timeline, Workload, and Activity-style perspectives.
- Custom Fields can be applied at Workspace, Space, Folder, or List level and can power reporting and automations.
- Automations use triggers, optional conditions, and actions. They can apply at List, Folder, or Space scope.
- Dashboards convert task and time data into reporting cards for progress, workload, project health, and OKRs.

Sources used for this baseline:

- ClickUp Hierarchy overview: https://help.clickup.com/hc/en-us/articles/13856392825367-Intro-to-the-Hierarchy
- ClickUp Folders and Lists: https://help.clickup.com/hc/en-us/articles/6311450560407-Learn-the-difference-between-Folders-and-Lists
- ClickUp Tasks overview: https://help.clickup.com/hc/en-us/articles/10552031987735-Task-View-3-0-overview
- ClickUp Views overview: https://help.clickup.com/hc/en-us/articles/6310383076503-Task-views
- ClickUp Custom Fields overview: https://help.clickup.com/hc/en-us/articles/6303536766231-Intro-to-Custom-Fields
- ClickUp Automations overview: https://help.clickup.com/hc/en-us/articles/6312102752791-Intro-to-Automations
- ClickUp Dashboards overview: https://help.clickup.com/hc/en-us/articles/6312197753239-Dashboards-overview

## Product Principles

- Hierarchy is not decoration. Folder, List, Task, and Subtask must each have distinct behavior and reporting value.
- Grant operations are the first proof case. If we can manage a real Toronto Arts Council application from discovery through budget submission, the system is useful.
- Agents are first-class collaborators. Grant Manager, Grant Writer, Budget Manager, Project Manager, and Grant Memory should be assignable, mentionable, and automatable.
- Views are lenses over the same work, not separate data silos.
- Every automation must be visible, reversible where possible, and auditable.
- Use conservative defaults: private grant folders, explicit assignees, visible due dates, and clear status transitions.
- Avoid generic project-management sprawl. Every field, view, and automation should help a real workflow.

## Canonical Hierarchy Model

### Workspace

The top-level organization that owns users, teams, permissions, global custom fields, global automations, audit logs, and cross-workspace search.

Initial example:

- `Street Voices / Nanobot`

Required capabilities:

- Workspace settings
- Global roles and teams
- Global custom fields
- Global task types
- Global templates
- Cross-space search
- Everything / All Tasks view
- Workspace dashboards
- Audit log

### Space

A major operating area, department, or durable workflow.

Initial examples:

- `Operations`
- `Grant Writing`
- `Finance`
- `Content`
- `Development`
- `Research`

Required capabilities:

- Space-level statuses
- Space-level custom fields
- Space-level required views
- Space-level dashboards
- Space members and permissions
- Space templates
- Space automation scope

### Folder

An optional grouping layer for related Lists. Folders do not directly contain tasks.

Initial example:

- `Grant`

Required capabilities:

- Folder settings
- Folder permissions
- Folder status
- Folder custom fields
- Folder-level views that aggregate child Lists
- Folder dashboards
- Folder templates
- Folder automation scope across all child Lists

### List

A concrete work container. Lists own tasks, default statuses, visible custom fields, templates, and primary views.

Initial example:

- `To do grants`

Required capabilities:

- List settings
- List statuses
- List custom fields
- List views
- List task templates
- List automations
- List-level permissions
- Sorting, grouping, filtering, and saved view state

### Task

A primary work item with enough structure to drive assignment, discussion, planning, and reporting.

Initial example:

- `Toronto Arts Council`

Required capabilities:

- Title
- Description
- Status
- Assignees
- Watchers / followers
- Priority
- Start date
- Due date
- Tags
- Task type
- Custom fields
- Relationships and dependencies
- Comments
- Assigned comments
- Attachments
- Docs
- Checklists
- Subtasks
- Time estimate
- Time tracked
- Activity log
- AI summary
- Home List plus additional Lists where needed

### Subtask

A child work item that can be assigned, dated, commented on, automated, and reported on like a task while preserving parent context.

Initial example:

- `Fill out the budget`

Required capabilities:

- Same core fields as a task
- Parent task breadcrumb
- Optional nested subtasks
- Rollup into parent task progress
- Subtask-specific status
- Subtask-specific assignee
- Subtask dependencies
- Subtask templates

### Checklist Item

A lightweight step that does not need full task behavior.

Budget checklist examples:

- Confirm requested amount
- Confirm project period
- Add personnel costs
- Add artist fees
- Add venue or production costs
- Add marketing costs
- Add administration costs
- Write budget notes
- Reconcile total against funder maximum

## First Vertical Slice: Toronto Arts Council Budget

The first real scenario should prove the entire hierarchy:

```text
Folder: Grant
List: To do grants
Task: Toronto Arts Council
Subtask: Fill out the budget
```

### Task Template: Grant Application

Default task fields:

| Field | Type | Example |
| --- | --- | --- |
| Funder | Text | Toronto Arts Council |
| Program | Text | TBD |
| Grant stage | Status | Drafting |
| Deadline | Date | TBD |
| Requested amount | Currency | CAD TBD |
| Grant period | Date range | TBD |
| Fit score | Number | 1-5 |
| Strategic value | Dropdown | High |
| Portal URL | URL | TBD |
| Lead agent | Person / agent | Grant Manager |
| Budget owner | Person / agent | Budget Manager |
| Narrative owner | Person / agent | Grant Writer |
| Compliance risk | Dropdown | Low / Medium / High |

Default subtasks:

- Research eligibility
- Extract application questions
- Draft project description
- Fill out the budget
- Draft budget justification
- Gather attachments
- Internal review
- Final compliance check
- Submit application
- Record submission notes

### Subtask Template: Fill Out The Budget

Default fields:

| Field | Type | Example |
| --- | --- | --- |
| Budget status | Status | Not started |
| Budget total | Currency | CAD TBD |
| Personnel total | Currency | CAD TBD |
| Artist fees total | Currency | CAD TBD |
| Production total | Currency | CAD TBD |
| Marketing total | Currency | CAD TBD |
| Admin total | Currency | CAD TBD |
| Match required | Checkbox | false |
| Funder template received | Checkbox | false |
| Needs narrative cross-check | Checkbox | true |

Default checklist:

- Confirm funder maximum and minimum request.
- Confirm eligible and ineligible costs.
- Confirm project dates.
- Build line-item budget.
- Add calculation notes for each line item.
- Confirm every budgeted activity appears in the narrative.
- Confirm every narrative activity has budget coverage.
- Check that totals match the requested amount.
- Ask Grant Manager for review.

Default automation ideas:

- When `Toronto Arts Council` moves to `DRAFTING`, create the standard grant subtasks.
- When `Fill out the budget` is created, assign Budget Manager and set `Needs narrative cross-check` to true.
- When `Budget status` changes to `Ready for review`, notify Grant Manager and Grant Writer.
- When all grant subtasks are complete, move parent task to `REVIEW`.
- Seven days before deadline, post a deadline warning if budget is not complete.

## ClickUp-Parity Feature Backlog

### P0: Core Work Graph

- [ ] Workspace, Space, Folder, List, Task, Subtask, nested subtask entities.
- [ ] Enforce that Folders contain Lists and Lists contain tasks.
- [ ] Breadcrumbs from subtask to task to List to Folder to Space to Workspace.
- [ ] Task home List.
- [ ] Tasks in multiple Lists.
- [ ] Parent-child relationships for subtasks.
- [ ] Checklist items.
- [ ] Attachments.
- [ ] Comments and activity log.
- [ ] Basic permissions by hierarchy location.
- [ ] Soft delete and archive.

### P1: Task Fields And Workflows

- [ ] Custom statuses by Space, Folder, and List.
- [ ] Status categories: open, active, done, closed.
- [ ] Assignees and watchers.
- [ ] Priorities.
- [ ] Start dates and due dates.
- [ ] Tags.
- [ ] Task types.
- [ ] Time estimates.
- [ ] Time tracking.
- [ ] Recurring tasks.
- [ ] Required fields by List or task type.
- [ ] Field history.
- [ ] Parent progress rollup from subtasks and checklists.

### P2: Custom Fields

- [ ] Custom field definitions at Workspace, Space, Folder, and List scope.
- [ ] Field types: text, long text, number, currency, date, date range, checkbox, dropdown, multi-select, person, URL, email, phone, formula, progress, relation, rollup.
- [ ] Field visibility by task type.
- [ ] Required custom fields.
- [ ] Field inheritance rules from higher hierarchy levels.
- [ ] Custom field columns in List and Table views.
- [ ] Custom fields available to automations and dashboard filters.

### P3: Views

- [ ] List view.
- [ ] Board view grouped by status, assignee, priority, or custom field.
- [ ] Table view.
- [ ] Calendar view.
- [ ] Timeline view.
- [ ] Gantt view with dependencies.
- [ ] Workload view by assignee and capacity.
- [ ] Team view.
- [ ] Activity view.
- [ ] Everything / All Tasks view.
- [ ] Saved filters, sorting, grouping, and pinned views.
- [ ] Per-user and shared view preferences.

### P4: Dependencies And Relationships

- [ ] Blocking / waiting-on dependencies.
- [ ] Related tasks.
- [ ] Related Docs.
- [ ] Parent-child task rollups.
- [ ] Milestones.
- [ ] Critical path for Gantt.
- [ ] Dependency warnings when due dates conflict.
- [ ] Cross-List and cross-Folder relationships.

### P5: Templates

- [ ] Workspace templates.
- [ ] Space templates.
- [ ] Folder templates.
- [ ] List templates.
- [ ] Task templates.
- [ ] Subtask templates.
- [ ] Checklist templates.
- [ ] Grant application template.
- [ ] Grant budget template.
- [ ] Template versioning.

### P6: Automations

- [ ] Automation engine with trigger, condition, action model.
- [ ] Scope automations to List, Folder, Space, or Workspace.
- [ ] Triggers: task created, status changed, assignee changed, due date arrives, due date changes, custom field changes, priority changes, tag added, comment added, subtasks resolved, checklist resolved, dependency unblocked, time tracked.
- [ ] Conditions: field equals, assignee is, status is, due date before or after, task type is, tag exists, parent field value, user role.
- [ ] Actions: create task, create subtask, apply template, change status, assign user or agent, add watcher, set custom field, add comment, send channel message, create Doc, call webhook, launch agent.
- [ ] Automation run history.
- [ ] Failure handling and retry policy.
- [ ] Loop prevention.
- [ ] Manual run / test mode.

### P7: Collaboration

- [ ] Mentions for users, agents, tasks, Docs, Lists, and Folders.
- [ ] Assigned comments.
- [ ] Comment threads.
- [ ] Chat-to-task creation.
- [ ] Task-to-chat updates.
- [ ] Notifications with user preferences.
- [ ] Saved items.
- [ ] Share links with permission controls.
- [ ] Docs attached at Space, Folder, List, and Task level.
- [ ] Whiteboard or canvas references.

### P8: Dashboards And Reporting

- [ ] Dashboard entity.
- [ ] Dashboard cards for task counts, status distribution, deadlines, workload, time tracked, overdue work, budget totals, and grant pipeline.
- [ ] Folder portfolio cards.
- [ ] Grant pipeline dashboard.
- [ ] Budget readiness dashboard.
- [ ] Agent workload dashboard.
- [ ] Export dashboard card data.
- [ ] Scheduled dashboard summaries.

### P9: Agent-Native Work Management

- [ ] Assign tasks to agents.
- [ ] Mention agents in comments.
- [ ] Agent task inbox.
- [ ] Agent run state linked to task activity.
- [ ] Agent-created subtasks require visible attribution.
- [ ] Budget Manager can draft budget tables into a task Doc.
- [ ] Grant Writer can cross-check narrative against budget.
- [ ] Grant Manager can run pre-submission checklist.
- [ ] Grant Memory can attach prior funder context.
- [ ] Human approval gates for submission, deletion, external email, and portal actions.

### P10: Search, Import, Export, And Integrations

- [ ] Global search across hierarchy, tasks, comments, Docs, and attachments.
- [ ] Filters for status, assignee, due date, priority, tags, custom fields, task type, and hierarchy location.
- [ ] CSV import and export.
- [ ] Markdown export for grant files.
- [ ] Calendar integration.
- [ ] Email-to-task.
- [ ] Webhooks.
- [ ] Public API.
- [ ] ClickUp import path.
- [ ] Attachment storage integration.

### P11: Governance, Security, And Reliability

- [ ] Role-based access control.
- [ ] Private Spaces, Folders, Lists, tasks, and Docs.
- [ ] Audit log for field changes, automation runs, agent actions, and permission changes.
- [ ] Retention policy.
- [ ] Backups.
- [ ] Rate limits.
- [ ] Idempotency keys for automation actions.
- [ ] Observability for slow views, failed automations, and agent runs.
- [ ] Accessibility checks for all task views.

## Data Model Draft

Core entities:

```text
Workspace
  has many Spaces
  has many Users
  has many Teams
  has many CustomFieldDefinitions

Space
  belongs to Workspace
  has many Folders
  has many Lists
  has many Statuses
  has many Automations

Folder
  belongs to Space
  has many Lists
  has many Statuses
  has many CustomFieldDefinitions
  has many Automations

List
  belongs to Space
  optionally belongs to Folder
  has many Tasks
  has many Statuses
  has many CustomFieldDefinitions
  has many Views
  has many Automations

Task
  belongs to one home List
  can appear in many additional Lists
  can have many Subtasks
  can have many ChecklistItems
  can have many Comments
  can have many Attachments
  can have many CustomFieldValues
  can have many Relationships

Subtask
  is a Task with a parentTaskId
```

Minimum tables to add when implementation starts:

- `workspaces`
- `spaces`
- `folders`
- `lists`
- `tasks`
- `task_list_memberships`
- `task_statuses`
- `task_custom_field_definitions`
- `task_custom_field_values`
- `task_comments`
- `task_checklist_items`
- `task_relationships`
- `task_views`
- `task_automations`
- `task_automation_runs`
- `task_time_entries`
- `task_templates`
- `dashboard_cards`

## Grant Workflow Statuses

Recommended statuses for the `Grant` Folder:

- `IDENTIFIED`
- `EVALUATING`
- `PURSUING`
- `DRAFTING`
- `BUDGETING`
- `REVIEW`
- `SUBMITTED`
- `AWARDED`
- `DECLINED`
- `RESUBMIT`
- `ACTIVE`
- `CLOSED`

Recommended budget subtask statuses:

- `NOT_STARTED`
- `COLLECTING_CONSTRAINTS`
- `DRAFTING_BUDGET`
- `NEEDS_NARRATIVE_CHECK`
- `READY_FOR_REVIEW`
- `REVISIONS`
- `APPROVED`
- `BLOCKED`

## Required Grant Views

For `Folder: Grant`:

- `Pipeline`: Board grouped by Grant stage.
- `Deadlines`: List grouped by due date month.
- `Budget Risk`: Table filtered to budget incomplete or compliance risk medium/high.
- `Agent Workload`: Workload grouped by assignee or agent.
- `Submitted`: List of submitted grants with outcomes and reporting dates.

For `List: To do grants`:

- `Open Grants`: List sorted by deadline.
- `By Stage`: Board grouped by status.
- `Calendar`: Calendar by deadline.
- `Budget Table`: Table with requested amount, budget total, budget status, and owner.
- `Review Queue`: Filtered list where status is review or budget is ready for review.

## Implementation Milestones

### Milestone 1: Persist The Hierarchy

Outcome: Users can create and navigate Workspace -> Space -> Folder -> List -> Task -> Subtask.

Deliverables:

- Prisma schema for core hierarchy.
- API routes for hierarchy CRUD.
- Seed data for the grant example.
- Breadcrumb UI.
- Basic List view.

Acceptance test:

- Create `Folder: Grant`, `List: To do grants`, `Task: Toronto Arts Council`, and `Subtask: Fill out the budget`.
- Reload the app and confirm the hierarchy persists.
- Open the subtask and see the full breadcrumb path.

### Milestone 2: Make Tasks Useful

Outcome: Tasks and subtasks can carry the fields needed for real grant work.

Deliverables:

- Status, assignee, priority, start date, due date, tags, and description.
- Comments and activity events.
- Checklist items.
- Attachments.
- Task detail panel.

Acceptance test:

- Add budget checklist items to `Fill out the budget`.
- Assign Budget Manager.
- Set the parent task deadline.
- Confirm changes appear in activity history.

### Milestone 3: Custom Fields And Grant Templates

Outcome: Grant applications can be created from a reusable template with budget fields.

Deliverables:

- Custom field definitions and values.
- Grant application task template.
- Grant budget subtask template.
- Template application UI or API.

Acceptance test:

- Create a new grant task from template.
- Confirm all standard subtasks and budget custom fields appear.
- Confirm budget custom fields are available in List/Table view.

### Milestone 4: Multiple Views

Outcome: The same grant work can be managed as a list, board, calendar, and budget table.

Deliverables:

- Board view grouped by status.
- Calendar view by deadline.
- Table view with custom field columns.
- Saved views.

Acceptance test:

- Move `Toronto Arts Council` from `DRAFTING` to `BUDGETING` by dragging a card.
- See the same status update in List and Table views.

### Milestone 5: Automations And Agents

Outcome: Repetitive grant coordination is automated and agent-visible.

Deliverables:

- Automation engine.
- Initial grant automations.
- Agent assignment.
- Automation run log.
- Human approval gate for sensitive actions.

Acceptance test:

- Move a grant task to `DRAFTING`.
- Confirm standard subtasks are created.
- Confirm Budget Manager is assigned to `Fill out the budget`.
- Confirm automation run history shows what happened.

### Milestone 6: Dashboards

Outcome: Leadership can see grant pipeline health and budget readiness.

Deliverables:

- Dashboard cards for grant count by stage, upcoming deadlines, overdue tasks, budget completion, requested funding total, and agent workload.
- Folder dashboard for `Grant`.
- Workspace dashboard for all task work.

Acceptance test:

- Update budget status and requested amount.
- Confirm dashboard cards update.

### Milestone 7: ClickUp-Parity Power Layer

Outcome: The system covers advanced workflows expected from a serious work platform.

Deliverables:

- Dependencies and Gantt.
- Recurring tasks.
- Time tracking.
- Public API and webhooks.
- Import/export.
- Advanced permissions.
- Universal search.
- Mobile-ready responsive views.

Acceptance test:

- Import a grant pipeline CSV.
- Create dependencies between budget, narrative, and review subtasks.
- Export a complete grant task package as Markdown.

## Living Gap Tracker

Use this table to track feature parity over time.

| Area | Target | Status | Notes |
| --- | --- | --- | --- |
| Hierarchy | Workspace, Space, Folder, List, Task, Subtask | Not started | First implementation target |
| Views | List, Board, Table, Calendar, Gantt, Timeline, Workload | Not started | Start with List, Board, Table |
| Custom fields | Scoped definitions and values | Not started | Required for grants |
| Automations | Trigger, condition, action engine | Not started | Grant template automation first |
| Dashboards | Cards and saved dashboards | Not started | Grant dashboard first |
| Agents | Assignable and mentionable agents | Partial foundation | Grant agents already exist |
| Permissions | Hierarchy-level RBAC | Not started | Needed before external users |
| Search | Global search and filters | Not started | Use hierarchy filters early |
| Import/export | CSV, Markdown, API | Not started | Grant exports are high value |

## Open Product Questions

- Should `Grant Writing` be its own Space, or should `Grant` remain a Folder under `Operations`?
- Should tasks and subtasks share one database table with `parentTaskId`, or use separate tables?
- Do we need nested subtasks on day one, or can checklist items cover the first budget workflow?
- Should agent assignments use the existing `User.isAgent` model in `social/prisma/schema.prisma`?
- Should grant budgets live as structured custom fields, attached spreadsheet files, or both?
- Which task views belong in the Social app shell versus a separate project-management route?

## Next Build Recommendation

Start with Milestone 1 and seed the exact hierarchy:

```text
Folder: Grant
List: To do grants
Task: Toronto Arts Council
Subtask: Fill out the budget
```

That gives us a real, inspectable path for every future feature. Each new capability should be validated against this scenario before expanding to the rest of the app.
