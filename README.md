# Stdbuf

![CI][ci-url]
[![codecov][codecov-image]][codecov-url]

[ci-url]: https://github.com/dikuchan/stdbuf/actions/workflows/ci.yaml
[ci-image]: https://github.com/dikuchan/stdbuf/actions/workflows/ci.yaml/badge.svg

[codecov-url]: https://codecov.io/gh/dikuchan/stdbuf
[codecov-image]: https://codecov.io/gh/dikuchan/stdbuf/branch/master/graph/badge.svg?token=EWNC1RJZOK

Size and time aware deduplicated asynchronous buffer.

Inspired by [ClickHouse buffer engine](https://clickhouse.com/docs/en/engines/table-engines/special/buffer/). Used for
the same purpose.

### Usage

```python
import asyncio
import time

from stdbuf import Stdbuf


async def read(buf: Stdbuf):
    for i in range(16):
        await buf.put(i)
        await asyncio.sleep(0.5)


async def write(buf: Stdbuf):
    while True:
        start = time.perf_counter()
        # Returns at least every 2 seconds.
        # May return earlier if full.
        data = await buf.get()
        elapsed = time.perf_counter() - start
        assert len(data) <= 4
        assert elapsed <= 2 + 1e-2


async def main():
    with Stdbuf(4, 2, True) as buf:
        done, pending = await asyncio.wait({
            asyncio.create_task(read(buf)),
            asyncio.create_task(write(buf)),
        },
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
```