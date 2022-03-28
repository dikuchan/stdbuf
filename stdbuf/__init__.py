import asyncio
from types import TracebackType
from typing import (
    Generic,
    Hashable,
    List,
    Optional,
    Type,
    TypeVar,
)

T = TypeVar("T", bound=Hashable)


class Stdbuf(Generic[T]):
    def __init__(
        self,
        maxsize: int,
        maxtime: float,
        dedup: bool = False,
    ) -> None:
        """
        Size- and time-aware deduplicated asynchronous buffer.

        Either when size of the buffer exceeds `maxsize` or once in `maxtime` seconds
        buffer becomes unlocked and its data is returned in `get`.
        Until that, awaiting `get` is blocking.
        Use `get_nowait` to skip waiting.

        Note that even if the buffer remains empty, empty list will be flushed each
        `maxtime` seconds.

        :param maxsize: Maximum number of items to keep in buffer before flush.
        :param maxtime: Period in seconds between force flushes.
        :param dedup: Whether to deduplicate items in buffer.
        """
        if maxsize < 0:
            raise ValueError("Parameter `maxsize` must be greater than 0")
        if maxtime < 0:
            raise ValueError("Parameter `maxtime` must be greater than 0")

        self._buffer: List[T] = []

        self._maxsize = maxsize
        self._maxtime = maxtime
        self._dedup = dedup
        # Use lock to safely manipulate buffer in `get`.
        self._lock = asyncio.Lock()
        # Signal to the `get` method that one of the conditions is met, and it is time
        # to return the buffer's content.
        self._size_event = asyncio.Event()
        self._time_event = asyncio.Event()

        async def wait() -> None:
            while True:
                await asyncio.sleep(self._maxtime)
                self._time_event.set()

        self._task = asyncio.create_task(wait())

    def __enter__(self) -> "Stdbuf[T]":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        """
        Cancel inner timer task.

        If not stopped, asyncio task will be destroyed by the interpreter.
        """
        self._task.cancel()

    async def put(self, item: T) -> bool:
        """
        Put item in the buffer.

        :return: Whether item was inserted.
            Return `False` if the buffer is full or if deduplication is on and
            duplicate is in the buffer already.
        """
        if self._dedup and item in self._buffer:
            return False
        buffer_size = len(self._buffer)
        if buffer_size >= self._maxsize:
            return False
        async with self._lock:
            self._buffer.append(item)
        if buffer_size + 1 >= self._maxsize:
            self._size_event.set()

        return True

    async def _get(self) -> List[T]:
        self._size_event.clear()
        self._time_event.clear()
        async with self._lock:
            buffer = self._buffer.copy()
            self._buffer.clear()

        return buffer

    async def get(self) -> List[T]:
        """
        Get content of the buffer.

        Block if buffer is not yet ready.
        Return either if `maxsize` is reached or `maxtime` has passed.
        """
        _, pending = await asyncio.wait(
            {
                asyncio.create_task(self._size_event.wait()),
                asyncio.create_task(self._time_event.wait()),
            },
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Exactly one task is pending, other is done.
        pending.pop().cancel()
        buffer = await self._get()

        return buffer

    async def get_nowait(self) -> Optional[List[T]]:
        """
        Get content of the buffer without blocking.

        Return buffer's content immediately if ready.
        If not, return `None`.
        """
        if self._size_event.is_set() or self._time_event.is_set():
            buffer = await self._get()
            return buffer

        return None

    def empty(self) -> bool:
        return not self._buffer

    def size(self) -> int:
        """
        Current size of the buffer.
        """
        return len(self._buffer)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def maxtime(self) -> float:
        return self._maxsize
