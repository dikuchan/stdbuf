import asyncio
from types import TracebackType
from typing import (
    Generic,
    Hashable,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
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
        Size and time bounded asynchronous buffer with deduplication.

        Put items in buffer asynchronously with ``put``.

        Either when size of the buffer exceeds ``maxsize`` or time since the first
        put operation after successful ``get`` exceeds ``maxtime``, buffer is unlocked
        and its content is returned in ``get``.
        Until then, awaiting ``get`` blocks.

        It is advised to use separate ``asyncio`` tasks to put items and to get them.

        :param maxsize: Maximum number of items to keep in buffer.
        :param maxtime: Maximum time between first put and return.
        :param dedup: Whether to deduplicate items in buffer.
        """
        if maxsize <= 0:
            raise ValueError("Parameter `maxsize` must be greater than 0")
        if maxtime <= 0:
            raise ValueError("Parameter `maxtime` must be greater than 0")

        if dedup:
            # O(1) for ``in`` operation, O(n) when converting to list.
            self._add = lambda s, i: s.add(i)
            self._buffer: Union[Set[T], List[T]] = set()  # type: ignore
        else:
            # All for O(1) (in the best case).
            self._add = lambda s, i: s.append(i)
            self._buffer: Union[Set[T], List[T]] = []  # type: ignore

        self._maxsize = maxsize
        self._maxtime = maxtime
        self._dedup = dedup

        # Use thread lock to safely manipulate buffer.
        self._lock = asyncio.Lock()

        # Signal to the ``get`` method that ``maxsize`` is reached.
        self._size_event = asyncio.Event()
        # Signal to the ``get`` method that ``maxtime`` is reached.
        self._time_event = asyncio.Event()
        # Signal to timer that the first element is in the buffer.
        self._wait_event = asyncio.Event()
        # Signal to ``put`` and timer that buffer is unlocked.
        # Start over again.
        self._flush_event = asyncio.Event()

        # Whether to block ``put``.
        self._block = False

        # Launch the background task with timer.
        async def timer() -> None:
            while True:
                await self._wait_event.wait()
                try:
                    await asyncio.wait_for(
                        self._flush_event.wait(),
                        timeout=self._maxtime,
                    )
                except asyncio.TimeoutError:
                    self._time_event.set()

        self._timer_task = asyncio.create_task(timer())

    def __enter__(self) -> "Stdbuf[T]":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    async def put(self, item: T) -> bool:
        """
        Put an item in the buffer.

        :return: Whether the item was inserted.
            Return ``False`` if ``dedup`` is on and the item is duplicate.
        """
        if self._block:
            await self._flush_event.wait()

        async with self._lock:
            if self._dedup and item in self._buffer:
                return False

            self._add(self._buffer, item)

            buffer_size = len(self._buffer)
            if buffer_size == 1:
                self._wait_event.set()
            if buffer_size >= self._maxsize:
                self._size_event.set()
                self._block = True

        return True

    async def get(self) -> List[T]:
        """
        Get content of the buffer.

        Blocks if buffer is not yet ready.
        Return either if ``maxsize`` or ``maxtime`` is reached.
        """
        self._flush_event.clear()

        _, pending = await asyncio.wait(
            {
                asyncio.create_task(self._size_event.wait()),
                asyncio.create_task(self._time_event.wait()),
            },
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Either one, or zero tasks are done.
        if pending:
            pending.pop().cancel()

        async with self._lock:
            self._wait_event.clear()
            self._size_event.clear()
            self._time_event.clear()

            # Signal to stop the timer and unblock ``put``.
            self._flush_event.set()

            self._block = False

            buffer = self._buffer.copy()
            self._buffer.clear()

        return list(buffer)

    def empty(self) -> bool:
        """
        Whether the buffer is empty.
        """
        return not self._buffer

    def size(self) -> int:
        """
        Current size of the buffer.
        """
        return len(self._buffer)

    def close(self) -> None:
        """
        Cancel inner timer ``asyncio`` task.
        """
        self._timer_task.cancel()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def maxtime(self) -> float:
        return self._maxtime
