import asyncio
import time
from itertools import cycle
from typing import Callable, Coroutine

import pytest

from stdbuf import Stdbuf

PERF_TIME_DELTA = 1e-2


@pytest.mark.asyncio
async def test_simple() -> None:
    with Stdbuf(3, 2.0, dedup=True) as buf:
        assert await buf.put(1)
        assert await buf.put(2)
        assert await buf.put(3)
        assert await buf.get() == [1, 2, 3]
        start = time.perf_counter()
        assert await buf.put(4)
        assert buf.size() == 1
        assert await buf.get() == [4]
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0 + PERF_TIME_DELTA


@pytest.mark.asyncio
async def test_ops() -> None:
    with Stdbuf(2, 1.0, dedup=True) as buf:
        assert buf.maxsize == 2
        assert buf.maxtime - 1.0 < 1e-9
        assert buf.empty()
        assert await buf.put("a")
        assert not buf.empty()
        assert buf.size() == 1


@pytest.mark.skip
async def run(
    maxsize: int,
    maxtime: float,
    dedup: bool,
    consumer: Callable[[Stdbuf], Coroutine],
    producer: Callable[[Stdbuf], Coroutine],
    return_when: str = asyncio.FIRST_EXCEPTION,
) -> None:
    with Stdbuf(maxsize, maxtime, dedup) as buf:
        done, pending = await asyncio.wait(
            {
                asyncio.create_task(producer(buf)),
                asyncio.create_task(consumer(buf)),
            },
            return_when=return_when,
        )
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception():
                raise task.exception()


@pytest.mark.asyncio
async def test_maxsize() -> None:
    BUF_SIZE = 10  # noqa
    BUF_TIME = 1.0  # noqa
    N_ITEMS = 1000  # noqa

    items = []

    async def produce(buf: Stdbuf[int]):
        for x in range(N_ITEMS):
            await buf.put(x)
        await buf.put(-1)
        await asyncio.sleep(2 * BUF_TIME)

    async def consume(buf: Stdbuf[int]):
        while True:
            start = time.perf_counter()
            xs = await buf.get()
            elapsed = time.perf_counter() - start

            assert len(xs) <= BUF_SIZE
            assert elapsed < BUF_TIME + PERF_TIME_DELTA

            items.extend(xs)
            if -1 in xs:
                return

    await run(BUF_SIZE, BUF_TIME, False, consume, produce)

    assert sorted(items) == [-1] + list(range(N_ITEMS))


@pytest.mark.asyncio
async def test_maxtime() -> None:
    BUF_SIZE = 10  # noqa
    BUF_TIME = 1.0  # noqa

    items = []

    async def produce(buf: Stdbuf[float]):
        for _ in range(BUF_SIZE):
            x = time.perf_counter()
            await buf.put(x)
            await asyncio.sleep(2 * BUF_TIME / BUF_SIZE)
        await asyncio.sleep(2 * BUF_TIME)

        await buf.put(time.perf_counter())
        await buf.put(time.perf_counter())
        await asyncio.sleep(2 * BUF_TIME)

        await buf.put(time.perf_counter())
        await asyncio.sleep(2 * BUF_TIME)

        await buf.put(-1)
        await asyncio.sleep(2 * BUF_TIME)

    async def consume(buf: Stdbuf[float]):
        while True:
            xs = await buf.get()

            assert len(xs) <= BUF_SIZE
            assert max(xs) - min(xs) < BUF_TIME + PERF_TIME_DELTA

            items.extend(xs)
            if -1 in xs:
                return

    await run(BUF_SIZE, BUF_TIME, False, consume, produce)

    assert len(items) == BUF_SIZE + 4


@pytest.mark.asyncio
async def test_dedup() -> None:
    BUF_SIZE = 100  # noqa
    BUF_TIME = 1.0  # noqa
    N_ITEMS = 10000  # noqa

    items = []

    async def produce(buf: Stdbuf[str]):
        success = 0
        for char, _ in zip(cycle("Hello, world!"), range(N_ITEMS)):
            if await buf.put(char):
                success += 1

        assert success == len(set("Hello, world!"))

        await asyncio.sleep(2 * BUF_TIME)

    async def consume(buf: Stdbuf[str]):
        while True:
            start = time.perf_counter()
            chars = await buf.get()
            elapsed = time.perf_counter() - start

            assert elapsed < BUF_TIME + PERF_TIME_DELTA

            items.extend(chars)

    await run(BUF_SIZE, BUF_TIME, True, consume, produce, asyncio.FIRST_COMPLETED)

    assert len(items) == len(set("Hello, world!"))


@pytest.mark.asyncio
async def test_multiple_producer() -> None:
    BUF_SIZE = 100  # noqa
    BUF_TIME = 1.0  # noqa

    items = []

    async def produce(buf: Stdbuf[int], begin: int, end: int):
        for x in range(begin, end):
            assert await buf.put(x)
            await asyncio.sleep(BUF_TIME / 100)
        await asyncio.sleep(2 * BUF_TIME)

    async def consume(buf: Stdbuf[int]):
        while True:
            start = time.perf_counter()
            data = await buf.get()
            elapsed = time.perf_counter() - start

            assert len(data) <= BUF_SIZE
            assert elapsed < BUF_TIME + PERF_TIME_DELTA

            items.extend(data)

    async def run():
        await asyncio.gather(*producers)

    with Stdbuf(BUF_SIZE, BUF_TIME, True) as buf:
        producers = []
        ranges = [(1000 * x, 1000 * y) for x, y in zip(range(0, 10), range(1, 11))]
        for begin, end in ranges:
            producer = asyncio.create_task(produce(buf, begin, end))
            producers.append(producer)

        done, pending = await asyncio.wait(
            {
                asyncio.create_task(run()),
                asyncio.create_task(consume(buf)),
            },
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception():
                raise task.exception()

    assert sorted(items) == list(range(0, 10000))
