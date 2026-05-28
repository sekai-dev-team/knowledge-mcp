---
title: "Concurrency Patterns"
tags: [programming, concurrency, patterns]
date: 2025-04-05
---

## Producer-Consumer

The producer-consumer pattern decouples work generation from work processing using a queue.

```python
import asyncio

async def producer(queue, items):
    for item in items:
        await queue.put(item)
    await queue.put(None)  # Sentinel

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        process(item)
```

## Thread Pool

For CPU-bound or blocking I/O operations, use a thread pool to avoid blocking the event loop:

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(heavy_computation, items))
```

## Readers-Writer Lock

Multiple readers can access simultaneously, but writers get exclusive access. Useful for shared state that is read frequently but written rarely.

## Circuit Breaker

Prevents cascading failures by stopping calls to a failing service:

1. **Closed** — normal operation
2. **Open** — failures exceed threshold, calls fail fast
3. **Half-Open** — after timeout, probe with test request

See [[Python Async Programming]] for async-specific patterns.
