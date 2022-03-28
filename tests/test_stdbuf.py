import asyncio
import time

import pytest

from stdbuf import Stdbuf


@pytest.mark.asyncio
async def test_simple():
    with Stdbuf(3, 2.0, dedup=True) as buf:
        assert await buf.put(1)
        assert await buf.put(2)
        assert await buf.put(3)
        assert not await buf.put(1)
        assert await buf.get() == [1, 2, 3]
        start = time.perf_counter()
        assert await buf.put(4)
        assert not await buf.get_nowait()
        assert await buf.get() == [4]
        elapsed = time.perf_counter() - start
        assert (elapsed - 2.0) < 0.1


@pytest.mark.asyncio
async def test_producer_consumer():
    BUF_SIZE = 5  # noqa
    BUF_TIME = 1.0  # noqa

    async def produce(buf: Stdbuf[int]):
        for x in range(20):
            await buf.put(x)
            await asyncio.sleep(0.15)
        for x in range(20, 40):
            await buf.put(x)
            await asyncio.sleep(0.3)

    async def consume(buf: Stdbuf[int]):
        while True:
            start = time.perf_counter()
            xs = await buf.get()
            elapsed = time.perf_counter() - start
            if not xs:
                return
            if xs[-1] < 20:
                assert len(xs) == BUF_SIZE or elapsed < BUF_TIME + 1e-2
            else:
                assert len(xs) < BUF_SIZE and abs(elapsed - BUF_TIME) < 1e-2

    with Stdbuf(BUF_SIZE, BUF_TIME, False) as buf:
        done, pending = await asyncio.wait(
            {
                asyncio.create_task(consume(buf)),
                asyncio.create_task(produce(buf)),
            },
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception():
                raise task.exception()


@pytest.mark.asyncio
async def test_producer_consumer_periodical():
    BUF_SIZE = 10  # noqa
    BUF_TIME = 3.0  # noqa

    async def produce(buf: Stdbuf[int]):
        for x in range(10):
            await buf.put(x)
            await asyncio.sleep(0.1)
        await asyncio.sleep(2 * BUF_TIME)
        for x in range(10, 20):
            await buf.put(x)
            await asyncio.sleep(0.1)
        await asyncio.sleep(2 * BUF_TIME)
        await buf.put(-1)

    async def consume(buf: Stdbuf[int]):
        while True:
            start = time.perf_counter()
            xs = await buf.get()
            elapsed = time.perf_counter() - start
            assert elapsed < BUF_TIME + 1e-2
            assert len(xs) <= BUF_SIZE
            if not xs:
                # Empty buffer is OK.
                continue
            if xs[0] == -1:
                return

    with Stdbuf(BUF_SIZE, BUF_TIME, False) as buf:
        done, pending = await asyncio.wait(
            {
                asyncio.create_task(consume(buf)),
                asyncio.create_task(produce(buf)),
            },
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception():
                raise task.exception()


@pytest.mark.asyncio
async def test_explicit_stop():
    buf = Stdbuf(2, 0.5, dedup=True)
    assert await buf.put({"a": 1})
    assert await buf.put({"b": 2})
    assert not await buf.put({"c": 3})
    assert await buf.get() == [{"a": 1}, {"b": 2}]
    # Return buffer's content only if `maxsize` is reached.
    buf.close()
    assert await buf.put({"d": 4})
    time.sleep(1)
    assert not await buf.get_nowait()
    assert await buf.put({"e": 5})
    assert await buf.get() == [{"d": 4}, {"e": 5}]
