---
title: "Project Architecture Guidelines"
tags: [architecture, projects, guidelines]
date: 2025-03-10
---

## Microservices Principles

Each service should own its data and expose it through well-defined APIs. Services communicate over HTTP or message queues.

Key tenets:
- Single responsibility per service
- Data isolation (no shared databases)
- Independent deployability
- Failure isolation

## Repository Structure

### Monorepo with packages
```
project/
├── packages/
│   ├── core/        # Shared utilities
│   ├── api/         # HTTP API layer
│   └── worker/      # Background jobs
├── tests/
└── docker/
```

## API Design

Follow RESTful conventions:
- Use nouns for resources: `/users`, `/documents`
- HTTP methods for actions: GET, POST, PUT, DELETE
- Version via URL prefix: `/v1/`
- Consistent error responses: `{ "error": "message", "code": "ERR_001" }`

## Testing Strategy

| Level | Scope | Speed | Frequency |
|-------|-------|-------|-----------|
| Unit | Single function | ms | Every commit |
| Integration | Service boundaries | seconds | Every PR |
| E2E | Full system | minutes | Pre-deploy |

Related: [[Machine Learning Fundamentals]] for ML project specifics.
