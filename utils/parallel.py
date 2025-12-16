"""
Parallel Processing Utilities

Provides utilities for running operations in parallel with proper error handling
and concurrency limits to avoid overwhelming external services.
"""

import asyncio
import logging
from typing import Any, Callable, List, TypeVar, Awaitable
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def parallel_execute(
    tasks: List[Awaitable[T]],
    max_concurrency: int = 10,
    return_exceptions: bool = True
) -> List[T]:
    """
    Execute multiple async tasks in parallel with concurrency limit.

    Args:
        tasks: List of awaitable tasks
        max_concurrency: Maximum number of concurrent tasks
        return_exceptions: If True, return exceptions in results instead of raising

    Returns:
        List of results (or exceptions if return_exceptions=True)
    """
    if not tasks:
        return []

    if len(tasks) <= max_concurrency:
        # Small batch - run all at once
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    # Large batch - use semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async def run_with_semaphore(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    wrapped_tasks = [run_with_semaphore(task) for task in tasks]
    return await asyncio.gather(*wrapped_tasks, return_exceptions=return_exceptions)


async def parallel_map(
    func: Callable[..., Awaitable[T]],
    items: List[Any],
    max_concurrency: int = 10,
    return_exceptions: bool = True
) -> List[T]:
    """
    Apply an async function to items in parallel.

    Args:
        func: Async function to apply
        items: List of items to process
        max_concurrency: Maximum concurrent operations
        return_exceptions: If True, return exceptions in results

    Returns:
        List of results in same order as items

    Example:
        async def fetch_data(url):
            ...

        results = await parallel_map(fetch_data, urls, max_concurrency=5)
    """
    tasks = [func(item) for item in items]
    return await parallel_execute(tasks, max_concurrency, return_exceptions)


async def parallel_batch_geocode(
    geocode_func: Callable[[str], Awaitable[dict]],
    addresses: List[str],
    max_concurrency: int = 5
) -> List[dict]:
    """
    Geocode multiple addresses in parallel.

    Args:
        geocode_func: Async geocoding function
        addresses: List of addresses to geocode
        max_concurrency: Max concurrent geocode requests (limit to avoid rate limiting)

    Returns:
        List of geocode results
    """
    logger.info(f"Parallel geocoding {len(addresses)} addresses (max {max_concurrency} concurrent)")
    return await parallel_map(geocode_func, addresses, max_concurrency, return_exceptions=True)


async def parallel_fetch_demographics(
    demo_func: Callable[[float, float], Awaitable[dict]],
    locations: List[tuple],
    max_concurrency: int = 3
) -> List[dict]:
    """
    Fetch demographics for multiple locations in parallel.

    Args:
        demo_func: Async demographics function taking (lat, lon)
        locations: List of (latitude, longitude) tuples
        max_concurrency: Max concurrent requests (lower for heavy API calls)

    Returns:
        List of demographics results
    """
    logger.info(f"Parallel fetching demographics for {len(locations)} locations")

    async def fetch_for_location(loc: tuple) -> dict:
        return await demo_func(loc[0], loc[1])

    return await parallel_map(fetch_for_location, locations, max_concurrency, return_exceptions=True)


def run_in_background(func: Callable[..., Awaitable[Any]]):
    """
    Decorator to run an async function in the background without waiting.

    Usage:
        @run_in_background
        async def send_analytics(data):
            await analytics_client.send(data)

        # This returns immediately, function runs in background
        send_analytics(my_data)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        asyncio.create_task(func(*args, **kwargs))
    return wrapper


class ParallelBatchProcessor:
    """
    Process items in batches with parallel execution.

    Usage:
        processor = ParallelBatchProcessor(batch_size=10, max_concurrency=5)
        results = await processor.process(items, async_func)
    """

    def __init__(self, batch_size: int = 50, max_concurrency: int = 10):
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    async def process(
        self,
        items: List[Any],
        func: Callable[[Any], Awaitable[T]]
    ) -> List[T]:
        """
        Process items in batches with parallel execution.

        Args:
            items: Items to process
            func: Async function to apply to each item

        Returns:
            List of results
        """
        all_results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await parallel_map(
                func, batch, self.max_concurrency, return_exceptions=True
            )
            all_results.extend(batch_results)

            logger.debug(
                f"Processed batch {i // self.batch_size + 1}, "
                f"total: {len(all_results)}/{len(items)}"
            )

        return all_results
