from __future__ import annotations

import asyncio
import unittest

from server.config.shutdown import cancel_leftover_tasks


class TestCancelLeftoverTasks(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_leftover_times_out_instead_of_hanging(self) -> None:
        async def _stubborn() -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(3600)

        asyncio.create_task(_stubborn())
        await asyncio.sleep(0)
        await cancel_leftover_tasks(timeout=0.05)


if __name__ == "__main__":
    unittest.main()
