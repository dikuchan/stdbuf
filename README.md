# Stdbuf

[![CI][ci-image]][ci-url]
[![codecov][codecov-image]][codecov-url]

[ci-url]: https://github.com/dikuchan/stdbuf/actions/workflows/ci.yaml
[ci-image]: https://github.com/dikuchan/stdbuf/actions/workflows/ci.yaml/badge.svg

[codecov-url]: https://codecov.io/gh/dikuchan/stdbuf
[codecov-image]: https://codecov.io/gh/dikuchan/stdbuf/branch/master/graph/badge.svg?token=EWNC1RJZOK

Size and time bounded asynchronous buffer with deduplication. 

Buffer's content is flushed when either maximum size or time since
the first insertion reaches the specified limits.

Inspired by [ClickHouse buffer engine](https://clickhouse.com/docs/en/engines/table-engines/special/buffer/). Used for
the same purpose.

### Note

Multiple concurrent consumers are not supported.

### Usage

```python
import asyncio
import time

from stdbuf import Stdbuf


async def produce(buf: Stdbuf[int]):
    for i in range(2 ** 16):
        await buf.put(i)
        # Duplicates are ignored.
        await buf.put(i)


async def consume(buf: Stdbuf[int]):
    while True:
        start = time.perf_counter()
        # Get data at least every 2 seconds.
        # May return earlier if full.
        data = await buf.get()
        elapsed = time.perf_counter() - start
        assert len(data) <= 16
        assert elapsed <= 0.5 + 1e-2


async def main():
    with Stdbuf(16, 0.5, True) as buf:
        done, pending = await asyncio.wait({
            asyncio.create_task(produce(buf)),
            asyncio.create_task(consume(buf)),
        },
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
```