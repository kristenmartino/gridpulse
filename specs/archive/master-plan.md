# Software Development Master Plan
## Building with Claude AI · Jira · Git/GitHub · GCP

---

## Executive Summary

This document is the definitive playbook for building production-grade software from zero to launch using four pillars: **Claude AI** as your engineering co-pilot, **Jira** for project orchestration, **Git/GitHub** for version control and CI/CD, and **Google Cloud Platform (GCP)** for infrastructure. It is written from the perspective of someone who has shipped systems at scale and knows where teams actually get stuck.

---

## Phase 0 — Foundation & Architecture (Week 1–2)

### 0.1 Define the System

Before writing a single line of code, lock down these artifacts:

- **Product Requirements Document (PRD)** — Use Claude to draft and stress-test your PRD. Prompt it with your problem space and let it poke holes in your assumptions. Ask: *"What edge cases am I missing? What will break at 100x scale?"*
- **Architecture Decision Records (ADRs)** — For every major choice (monolith vs. microservices, SQL vs. NoSQL, REST vs. gRPC), write a one-page ADR. Claude can generate these from a prompt like: *"Write an ADR comparing Cloud Run vs. GKE for a service handling 10k RPM with bursty traffic."*
- **System Context Diagram** — Draw the boundary of your system. What are the external actors, integrations, and data flows? Claude can generate Mermaid diagrams for this.

### 0.2 Set Up Jira

**Project Structure:**

| Jira Entity     | Maps To                        | Example                                  |
|-----------------|--------------------------------|------------------------------------------|
| Epic            | Major feature or system domain | `AUTH — Authentication & Authorization`  |
| Story           | User-facing capability         | `As a user, I can reset my password`     |
| Task            | Engineering work unit          | `Set up Cloud SQL Postgres instance`     |
| Sub-task        | Atomic implementation step     | `Write migration for users table`        |
| Bug             | Defect                         | `Login fails on Safari with 2FA enabled` |

**Workflow Configuration:**

```
Backlog → Ready for Dev → In Progress → In Review → QA → Done
```

**Board Setup:**

1. Create a Scrum board with 2-week sprints.
2. Add custom fields: `Component` (frontend, backend, infra, data), `Effort Points` (Fibonacci), `Risk Level` (low/med/high).
3. Create filters for each team member's active work.
4. Set up automation rules: auto-assign reviewer when moved to "In Review," auto-transition to "Done" when PR merges.

**Labels to standardize:**

`tech-debt`, `security`, `performance`, `ux`, `infrastructure`, `breaking-change`, `needs-design`, `spike`

### 0.3 Set Up Git/GitHub

**Repository Strategy:**

For most teams, start with a **monorepo** unless you have strong organizational reasons to split:

```
myproject/
├── apps/
│   ├── web/              # Frontend (Next.js, React, etc.)
│   ├── api/              # Backend API service
│   └── worker/           # Background job processor
├── packages/
│   ├── shared/           # Shared types, utils, constants
│   ├── db/               # Database schemas, migrations, seed
│   └── config/           # Shared config (ESLint, TS, etc.)
├── infrastructure/
│   ├── terraform/        # IaC for GCP
│   ├── docker/           # Dockerfiles
│   └── k8s/              # Kubernetes manifests (if using GKE)
├── .github/
│   ├── workflows/        # CI/CD pipelines
│   ├── CODEOWNERS        # Enforce review ownership
│   └── pull_request_template.md
├── docs/
│   ├── adr/              # Architecture Decision Records
│   ├── runbooks/         # Operational runbooks
│   └── api/              # API documentation
└── scripts/              # Dev tooling, setup scripts
```

**Branch Strategy — GitHub Flow (keep it simple):**

```
main (protected, always deployable)
 └── feature/PROJ-123-add-auth
 └── fix/PROJ-456-login-safari
 └── chore/PROJ-789-update-deps
```

- `main` is production. Always green. Always deployable.
- All work happens on feature branches named `{type}/JIRA-ID-short-description`.
- Merge via squash-merge PRs. No merge commits cluttering history.
- Delete branches after merge.

**Branch Protection Rules (non-negotiable):**

- Require PR review (minimum 1 approval).
- Require status checks to pass (CI pipeline).
- Require branch to be up-to-date with `main`.
- Enforce CODEOWNERS for sensitive paths (`/infrastructure`, `/packages/db`).
- No force pushes to `main`.

**CODEOWNERS example:**

```
/infrastructure/     @platform-team
/packages/db/        @backend-lead
/apps/web/           @frontend-team
*.tf                 @infra-lead
```

### 0.4 Set Up GCP

**Project Organization:**

```
Organization
├── Folder: Production
│   └── Project: myapp-prod
├── Folder: Staging
│   └── Project: myapp-staging
├── Folder: Development
│   └── Project: myapp-dev
└── Folder: Shared
    └── Project: myapp-shared (Artifact Registry, DNS, etc.)
```

**Day-1 GCP Services to Provision:**

| Service                | Purpose                                    |
|------------------------|--------------------------------------------|
| Cloud Run or GKE       | Compute (start with Cloud Run, graduate to GKE if needed) |
| Cloud SQL (PostgreSQL) | Primary relational database                |
| Cloud Memorystore      | Redis for caching & sessions               |
| Cloud Storage (GCS)    | Object storage (uploads, assets, backups)  |
| Artifact Registry      | Docker image storage                       |
| Cloud CDN + Load Balancer | Edge caching and traffic management     |
| Secret Manager         | Secrets and credentials                    |
| Cloud Logging + Monitoring | Observability stack                    |
| Cloud IAM              | Access control and service accounts        |
| VPC + Private Service Connect | Network isolation                  |

**Infrastructure as Code — Terraform:**

Everything in GCP is provisioned via Terraform. No ClickOps. Ever.

```hcl
# infrastructure/terraform/main.tf
module "network" {
  source = "./modules/network"
  project_id = var.project_id
  region     = var.region
}

module "database" {
  source = "./modules/database"
  project_id  = var.project_id
  network_id  = module.network.vpc_id
  db_tier     = "db-custom-2-8192"
}

module "compute" {
  source = "./modules/compute"
  project_id    = var.project_id
  image         = var.container_image
  database_url  = module.database.connection_string
}
```

Use Terraform workspaces or separate state files per environment (dev/staging/prod).

---

## Phase 1 — Sprint Zero: CI/CD, Dev Environment, Core Scaffolding (Week 2–3)

### 1.1 CI/CD Pipeline (GitHub Actions → GCP)

**Pipeline Architecture:**

```
Push to feature branch
  → Lint + Type Check
  → Unit Tests
  → Build Docker Image
  → Push to Artifact Registry (tagged with SHA)

PR Merged to main
  → All above +
  → Deploy to Staging (Cloud Run)
  → Run Integration Tests against Staging
  → Smoke Tests

Manual Trigger or Tag
  → Promote Staging image to Production
  → Canary Rollout (10% → 50% → 100%)
  → Post-deploy health checks
```

**GitHub Actions Workflow (core):**

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck
      - run: npm test -- --coverage

  build-and-deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.SA_EMAIL }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: |
          gcloud builds submit \
            --tag $REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:$GITHUB_SHA
      - run: |
          gcloud run deploy $SERVICE \
            --image $REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:$GITHUB_SHA \
            --region $REGION
```

**Authentication:** Use Workload Identity Federation (no service account keys in GitHub secrets — ever).

### 1.2 Local Development Environment

Use Docker Compose to mirror production locally:

```yaml
# docker-compose.yml
services:
  api:
    build: ./apps/api
    ports: ["8080:8080"]
    env_file: .env.local
    depends_on: [db, redis]
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: myapp
      POSTGRES_PASSWORD: localdev
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
volumes:
  pgdata:
```

### 1.3 Using Claude as Your Engineering Co-Pilot

**Where Claude accelerates every phase:**

| Task                        | How to Use Claude                                                    |
|-----------------------------|----------------------------------------------------------------------|
| Architecture design         | Feed it your constraints, ask for tradeoff analysis and diagrams     |
| Code generation             | Claude Code in terminal for scaffolding, boilerplate, CRUD layers    |
| Code review                 | Paste diffs, ask for security issues, performance concerns, style    |
| Writing tests               | Give it a function, ask for unit + edge case tests                   |
| Database schema design      | Describe your domain, get normalized schemas + migration files       |
| Terraform modules           | Describe your infra needs, get production-ready HCL                  |
| API design                  | Describe endpoints, get OpenAPI specs                                |
| Debugging                   | Paste error + context, get root cause analysis                       |
| Documentation               | Generate ADRs, runbooks, API docs, README files                     |
| Jira ticket writing         | Describe a feature, get well-structured stories with acceptance criteria |
| Incident response           | Paste logs/alerts, get triage steps and runbook suggestions          |
| Performance optimization    | Share query plans, flamegraphs, or profiles for analysis             |

**Claude Code (Terminal Agent) Workflow:**

```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Use it for generating scaffolding
claude "Create a new Express.js API service with TypeScript, 
       Prisma ORM connected to PostgreSQL, health check endpoint, 
       structured logging with pino, and Dockerfile for Cloud Run"

# Use it for writing migrations
claude "Write a Prisma migration that adds a 'teams' table with 
       name, slug (unique), created_at, and a many-to-many 
       relationship with users via a team_members join table"

# Use it for debugging
claude "This Cloud Run service is returning 503s intermittently. 
       Here are the logs: [paste logs]. What's the likely cause 
       and how do I fix it?"
```

---

## Phase 2 — Core Feature Development (Weeks 3–10)

### 2.1 Sprint Cadence

| Activity              | When                | Duration    | Who              |
|-----------------------|---------------------|-------------|------------------|
| Sprint Planning       | Monday, Day 1       | 1.5 hours   | Full team        |
| Daily Standup         | Every morning       | 15 min max  | Full team        |
| Backlog Refinement    | Wednesday, mid-week | 1 hour      | PM + Tech Lead   |
| Sprint Review/Demo    | Last Friday         | 1 hour      | Full team + stakeholders |
| Retrospective         | Last Friday         | 45 min      | Full team        |

### 2.2 Development Workflow (The Inner Loop)

```
1.  Pick a Jira ticket → Move to "In Progress"
2.  Create a branch: git checkout -b feature/PROJ-123-user-auth
3.  Write code (use Claude for complex logic, tests, boilerplate)
4.  Write/update tests (aim for 80%+ coverage on new code)
5.  Run locally: docker compose up → test manually
6.  Commit with conventional commits: 
      git commit -m "feat(auth): add JWT token refresh endpoint [PROJ-123]"
7.  Push and open PR → auto-links to Jira
8.  CI runs → all checks green
9.  Reviewer approves → Squash merge
10. Auto-deploy to staging → Verify
11. Jira ticket auto-transitions to "Done"
```

**Conventional Commit Format:**

```
feat(scope): description [JIRA-ID]     # New feature
fix(scope): description [JIRA-ID]      # Bug fix
chore(scope): description              # Maintenance
docs(scope): description               # Documentation
refactor(scope): description           # Code refactor
perf(scope): description               # Performance improvement
test(scope): description               # Tests
ci(scope): description                 # CI/CD changes
```

### 2.3 Code Review Standards

Every PR must answer:

1. **Does it work?** — Tests pass, no regressions.
2. **Is it safe?** — No SQL injection, XSS, auth bypasses, secret leaks.
3. **Is it maintainable?** — Clear naming, reasonable complexity, documented decisions.
4. **Is it observable?** — Proper logging, error handling, metrics.
5. **Is it deployable?** — No breaking changes without a migration path.

**PR Template:**

```markdown
## What
Brief description of what this PR does.

## Why
Link to Jira ticket. Explain the business context.

## How
Technical approach. Call out any non-obvious decisions.

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests (if applicable)
- [ ] Manual testing steps

## Checklist
- [ ] No secrets in code
- [ ] Database migrations are reversible
- [ ] API changes are backward-compatible
- [ ] Logging added for new error paths
- [ ] Documentation updated
```

---

## Phase 3 — Observability, Security, and Hardening (Weeks 8–12)

### 3.1 Observability Stack on GCP

**The Three Pillars:**

| Pillar  | GCP Service            | What to Capture                              |
|---------|------------------------|----------------------------------------------|
| Logs    | Cloud Logging          | Structured JSON logs, request IDs, user context |
| Metrics | Cloud Monitoring       | Latency (p50/p95/p99), error rate, saturation |
| Traces  | Cloud Trace            | Distributed tracing across services           |

**Key Alerts to Set Up (Day 1):**

- Error rate > 1% over 5 minutes.
- P95 latency > 500ms over 5 minutes.
- Cloud SQL CPU > 80% for 10 minutes.
- Cloud SQL connections > 80% of max.
- Cloud Run instance count hitting max.
- Any 5xx responses.
- Certificate expiry < 30 days.
- Failed deployments.

**Dashboards:**

Build a "Service Health" dashboard per service showing: request rate, error rate, latency distribution, active instances, CPU/memory utilization, and database connection pool status.

### 3.2 Security Hardening

**Non-negotiable security measures:**

- All secrets in Secret Manager (never in env vars, code, or Terraform state).
- Workload Identity Federation for CI/CD (no long-lived keys).
- VPC Service Controls around sensitive APIs.
- Cloud Armor WAF in front of public endpoints.
- Least-privilege IAM — every service account gets only what it needs.
- Dependency scanning via Dependabot or Renovate.
- Container scanning in Artifact Registry.
- HTTPS everywhere. HSTS headers. CSP headers.
- Auth tokens short-lived (15 min access, 7 day refresh).

### 3.3 Database Operations

- **Migrations:** Always forward-compatible. Never drop columns in the same release that stops using them. Use a two-phase approach: (1) deploy code that doesn't use the column, (2) next release drops the column.
- **Backups:** Cloud SQL automated backups daily, point-in-time recovery enabled, cross-region backup for production.
- **Connection pooling:** Use PgBouncer sidecar or Cloud SQL Auth Proxy with connection limits.
- **Query performance:** Set up slow query logging (> 100ms), use `pg_stat_statements` for analysis.

---

## Phase 4 — Testing Strategy (Continuous)

### 4.1 Test Pyramid

```
        ╱  E2E  ╲          ← Few (5-10 critical user journeys)
       ╱ Integr. ╲         ← Moderate (API contracts, DB queries)
      ╱   Unit    ╲        ← Many (business logic, pure functions)
```

| Level       | Tool                    | Runs When      | Target Coverage |
|-------------|-------------------------|----------------|-----------------|
| Unit        | Jest / Vitest           | Every push     | 80%+            |
| Integration | Supertest + Testcontainers | Every PR    | Critical paths  |
| E2E         | Playwright / Cypress    | Pre-deploy     | Core journeys   |
| Load        | k6 / Locust             | Weekly / pre-launch | Capacity plan |

### 4.2 Using Claude for Test Generation

```
Prompt: "Here is my user authentication service [paste code]. 
Write comprehensive unit tests covering: happy path, invalid 
credentials, expired tokens, rate limiting, SQL injection 
attempts, and concurrent login from multiple devices."
```

Claude is exceptional at identifying edge cases you'd miss and generating test fixtures.

---

## Phase 5 — Pre-Launch & Launch (Weeks 11–14)

### 5.1 Pre-Launch Checklist

**Infrastructure:**
- [ ] Production Terraform applied and verified.
- [ ] DNS configured with appropriate TTLs.
- [ ] SSL/TLS certificates provisioned and auto-renewing.
- [ ] CDN configured and cache rules tested.
- [ ] Cloud Armor rules in place.
- [ ] Auto-scaling policies tested (load test to 3x expected traffic).

**Reliability:**
- [ ] Health check endpoints returning correctly.
- [ ] Graceful shutdown handling in all services.
- [ ] Circuit breakers on external dependencies.
- [ ] Retry policies with exponential backoff.
- [ ] Rate limiting on public APIs.
- [ ] Database connection pool tuned.

**Observability:**
- [ ] All dashboards built and tested.
- [ ] All critical alerts configured and tested.
- [ ] On-call rotation established with PagerDuty/Opsgenie.
- [ ] Runbooks written for top 10 failure scenarios.
- [ ] Log retention policies set.

**Security:**
- [ ] Penetration test completed (or scheduled).
- [ ] OWASP Top 10 addressed.
- [ ] Data encryption at rest and in transit.
- [ ] GDPR/compliance requirements met.
- [ ] Incident response plan documented.

**Rollback:**
- [ ] Rollback procedure tested (can deploy previous version in < 5 min).
- [ ] Database rollback plan tested (reversible migrations).
- [ ] Feature flags in place for high-risk features.

### 5.2 Launch Day

```
T-1 day:  Final staging verification. War room scheduled.
T-0:      Deploy to production (canary 10%).
T+5 min:  Verify health checks, error rates, latency.
T+15 min: Promote to 50%.
T+30 min: Promote to 100%.
T+1 hour: Declare launch. Monitor closely for 4 hours.
T+24 hrs: Post-launch review. Close launch Jira epic.
```

---

## Phase 6 — Post-Launch Operations (Ongoing)

### 6.1 Operational Cadence

| Activity            | Frequency | Purpose                                    |
|---------------------|-----------|--------------------------------------------|
| On-call rotation    | Weekly    | 24/7 coverage for production issues        |
| Incident reviews    | Per incident | Blameless post-mortems               |
| Dependency updates  | Weekly    | Renovate/Dependabot PRs reviewed and merged |
| Capacity review     | Monthly   | Review utilization, forecast growth        |
| Security audit      | Quarterly | Review IAM, scan for vulnerabilities       |
| Architecture review | Quarterly | ADRs for system evolution                  |
| Cost optimization   | Monthly   | Review GCP billing, right-size resources   |

### 6.2 Jira for Ongoing Work

Maintain three backlogs:

1. **Feature Backlog** — New capabilities, driven by product.
2. **Tech Debt Backlog** — Refactors, upgrades, performance work.
3. **Bug Backlog** — Defects, prioritized by severity.

Rule of thumb: allocate sprints as 70% features, 20% tech debt, 10% bugs (adjust based on system maturity).

### 6.3 Continuous Improvement

- Run retrospectives honestly. Track action items in Jira.
- Use Claude to analyze incident reports and suggest architectural improvements.
- Review deployment frequency, lead time, change failure rate, and mean time to recovery (the DORA metrics).
- Goal: deploy to production multiple times per day with confidence.

---

## Appendix A — GCP Cost Optimization

| Strategy                     | Savings Potential |
|------------------------------|-------------------|
| Cloud Run min instances = 0  | 40–60% on idle    |
| Committed Use Discounts (CUDs) | 25–55% on compute |
| Cloud SQL right-sizing       | 20–40%            |
| GCS lifecycle policies       | 30–50% on storage |
| Preemptible/Spot VMs for batch | 60–80%          |
| Review and delete unused resources monthly | Varies |

## Appendix B — Claude Prompt Library for Engineering

```
# Architecture Review
"Review this system architecture for a [type] application handling 
[scale]. Identify single points of failure, bottlenecks, and 
security concerns. Suggest improvements."

# Code Generation
"Generate a [language] implementation of [feature] following 
[pattern]. Include error handling, logging, input validation, 
and unit tests."

# Incident Triage
"Here are the symptoms: [describe]. Here are the recent changes: 
[list]. Here are the logs: [paste]. What is the most likely root 
cause and what should I check first?"

# Performance Analysis
"Here is a slow database query and its EXPLAIN plan: [paste]. 
Suggest optimizations including index changes, query rewrites, 
and caching strategies."

# Security Review
"Review this code for security vulnerabilities. Check for: 
injection attacks, authentication bypasses, data exposure, 
insecure defaults, and missing input validation."
```

## Appendix C — Key Jira-GitHub-GCP Integrations

| Integration                        | How                                              |
|------------------------------------|--------------------------------------------------|
| Jira ↔ GitHub                      | GitHub for Jira app — auto-links commits, PRs, branches to tickets |
| GitHub → GCP (CI/CD)              | GitHub Actions + Workload Identity Federation    |
| GitHub → Jira (auto-transition)   | Smart commits: `PROJ-123 #done` in commit message |
| GCP Alerts → Jira                 | Cloud Monitoring → Pub/Sub → Cloud Function → Jira API (auto-create bug tickets) |
| GCP Alerts → Slack/PagerDuty     | Cloud Monitoring notification channels            |
| Terraform → GCP                   | `google` provider with service account            |
| Claude Code → GitHub              | Use in terminal for PR creation, code generation  |

---

*This plan is a living document. Update it as your system evolves. The best architecture is the one that ships and can be changed.*
