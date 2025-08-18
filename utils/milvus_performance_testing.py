#!/usr/bin/env python3
"""
Milvus Router Performance Testing Script

This script tests various Milvus operations with increasing batch sizes to identify
performance bottlenecks and breaking points.

Usage:
    export MILVUS_ROUTER_URL="http://localhost:8000"
    export STORE_ID="1"
    python milvus_perf_test.py

Monitor with:
    top -p $(pgrep milvus)
    iostat -x 1
    tail -f /path/to/milvus/logs/milvus.log | grep -i "slow\|timeout\|error"
"""

import asyncio
import os
import time
import logging
import numpy as np
import json
import signal
import sys
from typing import List, Dict, Tuple
from datetime import datetime
import statistics
from dataclasses import dataclass, asdict
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"milvus_perf_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("milvus-perf-test")

# Import the AsyncMilvusRouterClient (assuming it's available)
try:
    from milvus_router_client import AsyncMilvusRouterClient
except ImportError:
    logger.error("Cannot import AsyncMilvusRouterClient. Make sure the module is available.")
    sys.exit(1)

@dataclass
class OperationResult:
    """Results from a single operation"""
    operation: str
    batch_size: int
    duration: float
    success: bool
    error: str = ""
    items_processed: int = 0
    throughput: float = 0.0  # items per second

@dataclass
class TestStats:
    """Statistics for a test run"""
    operation: str
    batch_sizes: List[int]
    durations: List[float]
    success_rates: List[float]
    throughputs: List[float]
    avg_duration: float
    p95_duration: float
    max_batch_size_successful: int

class MilvusPerformanceTester:
    """Comprehensive Milvus performance tester"""
    
    def __init__(self):
        # Configuration from environment
        self.router_url = os.environ.get("MILVUS_ROUTER_URL", "http://localhost:8000")
        self.store_id = int(os.environ.get("STORE_ID", "1"))
        self.embedding_dim = int(os.environ.get("EMBEDDING_DIM", "768"))
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "100"))
        self.timeout_seconds = float(os.environ.get("OPERATION_TIMEOUT", "30.0"))
        
        # Test configuration
        self.batch_sizes = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100]
        self.batch_sizes = [bs for bs in self.batch_sizes if bs <= self.max_batch_size]
        
        # Results storage
        self.results: List[OperationResult] = []
        self.stats: Dict[str, TestStats] = {}
        
        # Control flags
        self.running = True
        self.current_operation = ""
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Initialized MilvusPerformanceTester")
        logger.info(f"Router URL: {self.router_url}")
        logger.info(f"Store ID: {self.store_id}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Max batch size: {self.max_batch_size}")
        logger.info(f"Batch sizes to test: {self.batch_sizes}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def generate_random_embedding(self) -> List[float]:
        """Generate a random embedding vector"""
        return np.random.normal(0, 1, self.embedding_dim).astype(np.float32).tolist()
    
    def generate_dummy_data(self, batch_size: int) -> Dict:
        """Generate dummy data for batch operations"""
        return {
            'track_ids': list(range(1000, 1000 + batch_size)),
            'embeddings': [self.generate_random_embedding() for _ in range(batch_size)],
            'store_ids': [self.store_id] * batch_size,
            'camera_ids': [random.randint(1, 10) for _ in range(batch_size)],
            'timestamps': [int(time.time()) + i for i in range(batch_size)]
        }
    
    async def test_operation_with_timeout(self, operation_func, *args, **kwargs) -> Tuple[bool, float, str, int]:
        """Execute an operation with timeout and error handling"""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                operation_func(*args, **kwargs),
                timeout=self.timeout_seconds
            )
            duration = time.time() - start_time
            
            # Determine items processed based on result type
            items_processed = 0
            if isinstance(result, (list, tuple)):
                items_processed = len(result)
            elif isinstance(result, dict):
                if 'total_inserted' in result:
                    items_processed = result['total_inserted']
                elif 'results' in result:
                    items_processed = len(result['results'])
                elif 'features' in result:
                    items_processed = len(result['features'])
                else:
                    items_processed = 1
            elif isinstance(result, int):
                items_processed = result
            else:
                items_processed = 1 if result else 0
            
            return True, duration, "", items_processed
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return False, duration, f"Timeout after {self.timeout_seconds}s", 0
        except Exception as e:
            duration = time.time() - start_time
            return False, duration, str(e), 0
    
    async def test_search_embedding(self, batch_size: int = 1) -> OperationResult:
        """Test single embedding search"""
        self.current_operation = f"search_embedding (batch_size={batch_size})"
        
        query_embedding = self.generate_random_embedding()
        
        async with AsyncMilvusRouterClient(self.router_url, self.store_id) as client:
            success, duration, error, items = await self.test_operation_with_timeout(
                client.search_embedding,
                query_embedding=query_embedding,
                top_k=10,
                store_filter=self.store_id
            )
        
        throughput = items / duration if duration > 0 else 0
        return OperationResult(
            operation="search_embedding",
            batch_size=batch_size,
            duration=duration,
            success=success,
            error=error,
            items_processed=items,
            throughput=throughput
        )
    
    async def test_search_embeddings_batch(self, batch_size: int) -> OperationResult:
        """Test batch embedding search"""
        self.current_operation = f"search_embeddings_batch (batch_size={batch_size})"
        
        embeddings_list = [self.generate_random_embedding() for _ in range(batch_size)]
        
        async with AsyncMilvusRouterClient(self.router_url, self.store_id) as client:
            success, duration, error, items = await self.test_operation_with_timeout(
                client.search_embeddings_batch,
                embeddings_list=embeddings_list,
                top_k=10,
                store_id=self.store_id
            )
        
        throughput = batch_size / duration if duration > 0 else 0
        return OperationResult(
            operation="search_embeddings_batch",
            batch_size=batch_size,
            duration=duration,
            success=success,
            error=error,
            items_processed=items,
            throughput=throughput
        )
    
    async def test_insert_embeddings_batch(self, batch_size: int) -> OperationResult:
        """Test batch embedding insertion"""
        self.current_operation = f"insert_embeddings_batch (batch_size={batch_size})"
        
        dummy_data = self.generate_dummy_data(batch_size)
        
        async with AsyncMilvusRouterClient(self.router_url, self.store_id) as client:
            success, duration, error, items = await self.test_operation_with_timeout(
                client.insert_embeddings_batch,
                track_ids=dummy_data['track_ids'],
                embeddings=dummy_data['embeddings'],
                store_ids=dummy_data['store_ids'],
                camera_ids=dummy_data['camera_ids'],
                timestamps=dummy_data['timestamps']
            )
        
        throughput = batch_size / duration if duration > 0 else 0
        return OperationResult(
            operation="insert_embeddings_batch",
            batch_size=batch_size,
            duration=duration,
            success=success,
            error=error,
            items_processed=items,
            throughput=throughput
        )
    
    async def test_get_features_track(self, track_id: int) -> OperationResult:
        """Test getting features for a specific track"""
        self.current_operation = f"get_features_track (track_id={track_id})"
        
        async with AsyncMilvusRouterClient(self.router_url, self.store_id) as client:
            success, duration, error, items = await self.test_operation_with_timeout(
                client.get_features_by_track_id,
                track_id=track_id,
                store_id=self.store_id
            )
        
        throughput = items / duration if duration > 0 else 0
        return OperationResult(
            operation="get_features_track",
            batch_size=1,
            duration=duration,
            success=success,
            error=error,
            items_processed=items,
            throughput=throughput
        )
    
    def log_result(self, result: OperationResult):
        """Log the result of an operation"""
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        
        logger.info(
            f"{status} | {result.operation:25} | "
            f"batch={result.batch_size:3d} | "
            f"time={result.duration:7.3f}s | "
            f"throughput={result.throughput:8.2f} items/s | "
            f"processed={result.items_processed:4d}"
        )
        
        if not result.success:
            logger.error(f"  Error: {result.error}")
    
    def calculate_stats(self, operation: str) -> TestStats:
        """Calculate statistics for an operation"""
        op_results = [r for r in self.results if r.operation == operation]
        if not op_results:
            return None
        
        batch_sizes = [r.batch_size for r in op_results]
        durations = [r.duration for r in op_results]
        success_rates = []
        throughputs = [r.throughput for r in op_results if r.success]
        
        # Calculate success rate per batch size
        for bs in set(batch_sizes):
            bs_results = [r for r in op_results if r.batch_size == bs]
            success_rate = sum(1 for r in bs_results if r.success) / len(bs_results) * 100
            success_rates.append(success_rate)
        
        successful_results = [r for r in op_results if r.success]
        max_successful_batch = max([r.batch_size for r in successful_results]) if successful_results else 0
        
        return TestStats(
            operation=operation,
            batch_sizes=list(set(batch_sizes)),
            durations=durations,
            success_rates=success_rates,
            throughputs=throughputs,
            avg_duration=statistics.mean(durations) if durations else 0,
            p95_duration=statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations) if durations else 0,
            max_batch_size_successful=max_successful_batch
        )
    
    def print_summary(self):
        """Print a comprehensive summary of all test results"""
        logger.info("=" * 80)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("=" * 80)
        
        operations = list(set(r.operation for r in self.results))
        
        for operation in operations:
            stats = self.calculate_stats(operation)
            if not stats:
                continue
                
            logger.info(f"\n{operation.upper()}")
            logger.info("-" * 50)
            logger.info(f"  Average duration: {stats.avg_duration:.3f}s")
            logger.info(f"  P95 duration: {stats.p95_duration:.3f}s")
            logger.info(f"  Max successful batch size: {stats.max_batch_size_successful}")
            
            if stats.throughputs:
                logger.info(f"  Average throughput: {statistics.mean(stats.throughputs):.2f} items/s")
                logger.info(f"  Max throughput: {max(stats.throughputs):.2f} items/s")
            
            # Show batch size breakdown
            batch_results = defaultdict(list)
            for result in [r for r in self.results if r.operation == operation]:
                batch_results[result.batch_size].append(result)
            
            logger.info("  Batch size breakdown:")
            for batch_size in sorted(batch_results.keys()):
                results = batch_results[batch_size]
                success_count = sum(1 for r in results if r.success)
                avg_duration = statistics.mean([r.duration for r in results])
                logger.info(f"    Batch {batch_size:3d}: {success_count}/{len(results)} success, avg {avg_duration:.3f}s")
        
        # Save detailed results to JSON
        results_file = f"milvus_perf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        logger.info(f"\nDetailed results saved to: {results_file}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive performance tests"""
        logger.info("Starting Milvus Performance Testing...")
        logger.info("Use Ctrl+C to stop testing gracefully")
        
        test_sequence = [
            ("search_embedding", self.test_search_embedding),
            ("search_embeddings_batch", self.test_search_embeddings_batch),
            ("insert_embeddings_batch", self.test_insert_embeddings_batch),
        ]
        
        try:
            # Test single operations first
            for operation_name, test_func in test_sequence:
                if not self.running:
                    break
                    
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing: {operation_name.upper()}")
                logger.info(f"{'='*60}")
                
                if operation_name == "search_embedding":
                    # Test single search multiple times
                    for i in range(5):
                        if not self.running:
                            break
                        result = await test_func(1)
                        self.results.append(result)
                        self.log_result(result)
                        await asyncio.sleep(1)  # Brief pause between tests
                else:
                    # Test with increasing batch sizes
                    for batch_size in self.batch_sizes:
                        if not self.running:
                            break
                            
                        logger.info(f"\nTesting batch size: {batch_size}")
                        result = await test_func(batch_size)
                        self.results.append(result)
                        self.log_result(result)
                        
                        # If operation failed, try smaller increases
                        if not result.success:
                            logger.warning(f"Operation failed at batch size {batch_size}")
                            if "timeout" in result.error.lower():
                                logger.warning("Timeout detected - database may be overwhelmed")
                            break
                        
                        # Brief pause between tests to avoid overwhelming the system
                        await asyncio.sleep(2)
            
            # Test get_features_track for a few different track IDs
            if self.running:
                logger.info(f"\n{'='*60}")
                logger.info("Testing: GET_FEATURES_TRACK")
                logger.info(f"{'='*60}")
                
                test_track_ids = [1, 2, 100, 1000]  # Test various track IDs
                for track_id in test_track_ids:
                    if not self.running:
                        break
                    result = await self.test_get_features_track(track_id)
                    self.results.append(result)
                    self.log_result(result)
                    await asyncio.sleep(1)
            
            # Continuous testing mode (until interrupted)
            if self.running:
                logger.info(f"\n{'='*60}")
                logger.info("Starting continuous testing mode...")
                logger.info("Use Ctrl+C to stop and see final results")
                logger.info(f"{'='*60}")
                
                cycle_count = 0
                while self.running:
                    cycle_count += 1
                    logger.info(f"\n--- Continuous Test Cycle {cycle_count} ---")
                    
                    # Run a mixed workload
                    operations = [
                        ("search_embedding", lambda: self.test_search_embedding(1)),
                        ("search_batch_small", lambda: self.test_search_embeddings_batch(5)),
                        ("insert_batch_small", lambda: self.test_insert_embeddings_batch(10)),
                        ("get_features", lambda: self.test_get_features_track(random.randint(1, 100)))
                    ]
                    
                    for op_name, op_func in operations:
                        if not self.running:
                            break
                        result = await op_func()
                        self.results.append(result)
                        self.log_result(result)
                        await asyncio.sleep(0.5)  # Short pause between operations
                    
                    await asyncio.sleep(5)  # Pause between cycles
        
        except Exception as e:
            logger.error(f"Test execution error: {e}")
        
        finally:
            logger.info("\nTest execution completed or interrupted")
            self.print_summary()

async def main():
    """Main entry point"""
    # Validate environment variables
    required_vars = ["MILVUS_ROUTER_URL"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set: export MILVUS_ROUTER_URL='http://your-router:8000'")
        sys.exit(1)
    
    tester = MilvusPerformanceTester()
    
    logger.info("Milvus Performance Testing Script")
    logger.info("=" * 50)
    logger.info("This script will test Milvus operations with increasing batch sizes")
    logger.info("Monitor your Milvus instance with:")
    logger.info("  top -p $(pgrep milvus)")
    logger.info("  iostat -x 1")
    logger.info("  tail -f /path/to/milvus/logs/milvus.log | grep -i 'slow\\|timeout\\|error'")
    logger.info("=" * 50)
    
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
