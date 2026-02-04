#!/usr/bin/env python3
"""
Load Testing Script for Dhivehi Transliteration API
Simulates multiple concurrent users making transliteration requests
"""

import requests
import json
import time
import concurrent.futures
import statistics
from datetime import datetime

# Configuration
API_URL = "http://localhost:5001/transliterate"
TEST_TEXTS = [
    "kalhu kalhu meeheh viyya dhen",
    "kuriah othee LCE adhi WDC ah member activate kos",
    "mi aharu 17 february than kolheh bodukoh faahaga koshlan muhimmu kameh gothugai fennaany",
    "dhivehi raajje ge raees Mohamed Nasheed",
    "maaldhives ge fasveyo haalu",
]

def make_request(user_id, text):
    """Make a single transliteration request"""
    start_time = time.time()
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=300  # 5 minutes
        )

        # Read SSE stream
        result = None
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if not data.get('partial', True):  # Final result
                        result = data.get('thaana', '')

        end_time = time.time()
        duration = end_time - start_time

        return {
            'user_id': user_id,
            'success': True,
            'duration': duration,
            'result': result[:50] + '...' if result else None,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        return {
            'user_id': user_id,
            'success': False,
            'duration': duration,
            'error': str(e),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }

def run_load_test(num_users=10, is_concurrent=True):
    """
    Run load test with multiple concurrent users

    Args:
        num_users: Number of users to simulate
        is_concurrent: If True, run requests in parallel. If False, run sequentially.
    """
    print("="*80)
    print(f"🧪 LOAD TEST: {num_users} users - {'Concurrent' if is_concurrent else 'Sequential'}")
    print("="*80)
    print(f"API: {API_URL}")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)

    # Prepare test data
    test_data = [(i+1, TEST_TEXTS[i % len(TEST_TEXTS)]) for i in range(num_users)]

    overall_start = time.time()

    if is_concurrent:
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(make_request, user_id, text) for user_id, text in test_data]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        # Run sequential requests
        results = [make_request(user_id, text) for user_id, text in test_data]

    overall_end = time.time()
    overall_duration = overall_end - overall_start

    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    durations = [r['duration'] for r in successful]

    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)

    # Summary
    print(f"\n✅ Successful: {len(successful)}/{num_users}")
    print(f"❌ Failed: {len(failed)}/{num_users}")
    print(f"⏱️  Total time: {overall_duration:.2f}s")

    if durations:
        print(f"\n📈 Response Times:")
        print(f"   Min:     {min(durations):.2f}s")
        print(f"   Max:     {max(durations):.2f}s")
        print(f"   Average: {statistics.mean(durations):.2f}s")
        print(f"   Median:  {statistics.median(durations):.2f}s")

        if is_concurrent:
            throughput = len(successful) / overall_duration
            print(f"\n🚀 Throughput: {throughput:.2f} requests/second")
            print(f"   (Handling {len(successful)} users in {overall_duration:.2f}s)")

    # Show individual results
    print(f"\n📝 Individual Results:")
    print(f"{'User':<6} {'Time':<10} {'Duration':<10} {'Status':<10} {'Result/Error'}")
    print("-"*80)

    for r in sorted(results, key=lambda x: x['user_id']):
        status = "✅ Success" if r['success'] else "❌ Failed"
        detail = r.get('result', r.get('error', 'Unknown'))[:40]
        print(f"{r['user_id']:<6} {r['timestamp']:<10} {r['duration']:>6.2f}s    {status:<10} {detail}")

    # Show failures in detail
    if failed:
        print(f"\n❌ FAILURES ({len(failed)}):")
        print("-"*80)
        for r in failed:
            print(f"User {r['user_id']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80 + "\n")

    return {
        'total_users': num_users,
        'successful': len(successful),
        'failed': len(failed),
        'total_duration': overall_duration,
        'durations': durations,
        'concurrent': is_concurrent
    }

if __name__ == "__main__":
    import sys

    # Check if server is running
    try:
        response = requests.get("http://localhost:5001/", timeout=5)
        print("✅ Server is running!\n")
    except:
        print("❌ ERROR: Server is not running!")
        print("   Start the server first:")
        print("   gunicorn -c gunicorn.conf.py app:app")
        print("   OR")
        print("   docker-compose up\n")
        sys.exit(1)

    # Parse command line arguments
    num_users = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("\n🎯 Running load tests...\n")

    # Test 1: Sequential (baseline)
    print("\n" + "🔄 Test 1: Sequential (one at a time)")
    sequential_results = run_load_test(num_users=5, is_concurrent=False)

    time.sleep(2)  # Brief pause between tests

    # Test 2: Concurrent (realistic)
    print("\n" + "⚡ Test 2: Concurrent (all at once)")
    concurrent_results = run_load_test(num_users=num_users, is_concurrent=True)

    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON")
    print("="*80)
    print(f"Sequential: {sequential_results['total_duration']:.2f}s for {sequential_results['total_users']} users")
    print(f"Concurrent: {concurrent_results['total_duration']:.2f}s for {concurrent_results['total_users']} users")

    if sequential_results['durations'] and concurrent_results['durations']:
        seq_avg = statistics.mean(sequential_results['durations'])
        conc_avg = statistics.mean(concurrent_results['durations'])
        print(f"\nAverage response time:")
        print(f"  Sequential: {seq_avg:.2f}s")
        print(f"  Concurrent: {conc_avg:.2f}s")

        speedup = sequential_results['total_duration'] / concurrent_results['total_duration']
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with concurrent processing")

    print("="*80 + "\n")
