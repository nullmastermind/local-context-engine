"""Cross-platform file lock for index build serialization."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Optional


class IndexLock:
    """A cross-platform file lock that prevents concurrent index builds.

    Uses ``fcntl.flock`` on Unix and ``msvcrt.locking`` on Windows.
    Falls back to a polling loop so callers don't need to worry about
    platform differences.

    Usage::

        lock = IndexLock(db_path.parent / "index.lock")
        with lock:
            # only one process at a time reaches here
            ...

    Args:
        lock_path: Path to the lock file (created if absent).
        timeout: Maximum seconds to wait before raising TimeoutError.
    """

    def __init__(self, lock_path: Path | str, timeout: int = 300) -> None:
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self._fh: Optional[object] = None  # file handle, platform-specific type

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def acquire(self) -> None:
        """Block until the lock is acquired or timeout is reached.

        Raises:
            TimeoutError: If the lock cannot be acquired within ``timeout`` seconds.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout

        while True:
            try:
                self._acquire_once()
                return  # success
            except (OSError, IOError):
                pass  # lock held by another process

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Could not acquire index lock at {self.lock_path} "
                    f"within {self.timeout}s. Another build may be running."
                )
            time.sleep(0.25)

    def release(self) -> None:
        """Release the lock and close the file handle.

        Safe to call even if the lock was never acquired.
        """
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                self._release_windows()
            else:
                self._release_unix()
        except Exception:
            pass
        finally:
            try:
                self._fh.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._fh = None

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "IndexLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.release()

    # ------------------------------------------------------------------ #
    # Platform-specific internals                                          #
    # ------------------------------------------------------------------ #

    def _acquire_once(self) -> None:
        """Attempt a single non-blocking lock acquisition.

        Raises OSError/IOError if the lock is held by another process.
        """
        if sys.platform == "win32":
            self._acquire_windows()
        else:
            self._acquire_unix()

    def _acquire_unix(self) -> None:
        import fcntl

        fh = open(self.lock_path, "w")  # noqa: WPS515
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            fh.close()
            raise
        self._fh = fh

    def _release_unix(self) -> None:
        import fcntl

        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]

    def _acquire_windows(self) -> None:
        import msvcrt

        fh = open(self.lock_path, "w")  # noqa: WPS515
        try:
            # Lock the first byte of the file (non-blocking)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except (OSError, IOError):
            fh.close()
            raise
        self._fh = fh

    def _release_windows(self) -> None:
        import msvcrt

        try:
            # Seek back to start before unlocking
            self._fh.seek(0)  # type: ignore[union-attr]
        except Exception:
            pass
        msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[union-attr]
