from __future__ import annotations

import asyncio


async def gather_requests(*operations):
    """Settle every admitted request before returning or propagating an error."""
    results = await asyncio.gather(*operations, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results
