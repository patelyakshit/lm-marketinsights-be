"""
Unit tests for Parallel Processing module.
"""

import asyncio
import pytest
from utils.parallel import (
    parallel_execute,
    parallel_map,
    parallel_batch_geocode,
    parallel_fetch_demographics,
    ParallelBatchProcessor,
    run_in_background,
)


class TestParallelExecute:
    """Tests for parallel_execute function."""

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        """Test with empty task list."""
        result = await parallel_execute([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_task(self):
        """Test with single task."""
        async def task():
            return "result"

        result = await parallel_execute([task()])
        assert result == ["result"]

    @pytest.mark.asyncio
    async def test_multiple_tasks(self):
        """Test with multiple tasks."""
        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2

        tasks = [task(i) for i in range(5)]
        results = await parallel_execute(tasks)

        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        concurrent_count = 0
        max_concurrent = 0

        async def counting_task():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return True

        tasks = [counting_task() for _ in range(10)]
        await parallel_execute(tasks, max_concurrency=3)

        # Max concurrent should not exceed limit
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_return_exceptions(self):
        """Test exception handling."""
        async def failing_task():
            raise ValueError("Test error")

        async def passing_task():
            return "ok"

        tasks = [passing_task(), failing_task(), passing_task()]
        results = await parallel_execute(tasks, return_exceptions=True)

        assert len(results) == 3
        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok"


class TestParallelMap:
    """Tests for parallel_map function."""

    @pytest.mark.asyncio
    async def test_map_function(self):
        """Test mapping function over items."""
        async def double(x):
            return x * 2

        results = await parallel_map(double, [1, 2, 3, 4, 5])
        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_map_preserves_order(self):
        """Test that results preserve input order."""
        async def identity_with_delay(x):
            # Add variable delay to test ordering
            await asyncio.sleep(0.01 * (5 - x))
            return x

        results = await parallel_map(identity_with_delay, [1, 2, 3, 4, 5])
        assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_map_empty_list(self):
        """Test mapping over empty list."""
        async def double(x):
            return x * 2

        results = await parallel_map(double, [])
        assert results == []


class TestParallelBatchProcessor:
    """Tests for ParallelBatchProcessor class."""

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test processing items in batches."""
        processor = ParallelBatchProcessor(batch_size=3, max_concurrency=2)

        async def process_item(x):
            return x ** 2

        items = list(range(10))
        results = await processor.process(items, process_item)

        assert len(results) == 10
        assert results == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    @pytest.mark.asyncio
    async def test_batch_size_respected(self):
        """Test that batch size is respected."""
        batches_processed = []

        async def track_batch(x):
            # Track which items are processed together
            batches_processed.append(x)
            await asyncio.sleep(0.01)
            return x

        processor = ParallelBatchProcessor(batch_size=5, max_concurrency=10)
        items = list(range(12))
        await processor.process(items, track_batch)

        # All items should be processed
        assert sorted(batches_processed) == items


class TestParallelBatchGeocode:
    """Tests for parallel batch geocoding."""

    @pytest.mark.asyncio
    async def test_batch_geocode_empty(self):
        """Test with empty address list."""
        async def mock_geocode(address):
            return {"address": address, "lat": 0, "lng": 0}

        results = await parallel_batch_geocode(mock_geocode, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_geocode_multiple(self):
        """Test geocoding multiple addresses."""
        async def mock_geocode(address):
            return {"address": address, "found": True}

        addresses = ["Address 1", "Address 2", "Address 3"]
        results = await parallel_batch_geocode(mock_geocode, addresses, max_concurrency=2)

        assert len(results) == 3
        assert all(r["found"] for r in results)


class TestParallelFetchDemographics:
    """Tests for parallel demographics fetching."""

    @pytest.mark.asyncio
    async def test_fetch_demographics_empty(self):
        """Test with empty location list."""
        async def mock_demo(lat, lon):
            return {"lat": lat, "lon": lon}

        results = await parallel_fetch_demographics(mock_demo, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_demographics_multiple(self):
        """Test fetching demographics for multiple locations."""
        async def mock_demo(lat, lon):
            return {"lat": lat, "lon": lon, "population": 1000}

        locations = [(32.78, -96.80), (40.71, -74.01), (34.05, -118.24)]
        results = await parallel_fetch_demographics(mock_demo, locations, max_concurrency=2)

        assert len(results) == 3
        assert all(r["population"] == 1000 for r in results)


class TestRunInBackground:
    """Tests for run_in_background decorator."""

    @pytest.mark.asyncio
    async def test_background_execution(self):
        """Test that function runs in background."""
        result_holder = {"completed": False}

        @run_in_background
        async def background_task():
            await asyncio.sleep(0.1)
            result_holder["completed"] = True

        # Call the decorated function
        background_task()

        # Should return immediately (not complete yet)
        assert not result_holder["completed"]

        # Wait for background task
        await asyncio.sleep(0.2)

        # Now should be complete
        assert result_holder["completed"]
