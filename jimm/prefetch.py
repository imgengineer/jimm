"""Host-to-device batch prefetching shared by the training and evaluation loops."""

import collections
import contextlib

__all__ = ["prefetch_stream"]


def prefetch_stream(source, put, prefetch_size=2):
    """Yield ``put(batch)`` results while keeping ``prefetch_size`` batches in flight.

    MaxText/MaxDiffusion double buffering: ``put`` runs for the next batch before the
    current one is handed to the caller, so host-to-device transfers overlap compute.
    ``prefetch_size <= 0`` yields every batch without buffering.
    """
    iterator = iter(source)
    if prefetch_size <= 0:
        for batch in iterator:
            yield put(batch)
        return
    queue = collections.deque()
    for _ in range(prefetch_size):
        try:
            queue.append(put(next(iterator)))
        except StopIteration:
            break
    while queue:
        item = queue.popleft()
        with contextlib.suppress(StopIteration):
            queue.append(put(next(iterator)))
        yield item
