
import time
import statistics
import json
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000/predict"
BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "reports" / "benchmark_results.json"

payload = {
    "Time": 1000,
    **{f"V{i}": 0.1 for i in range(1, 29)},
    "Amount": 50.0,
}

times = []

print("Starting FraudShield API benchmark...")

with httpx.Client() as client:
    response = client.post(URL, json=payload)
    response.raise_for_status()

    for _ in range(100):
        start = time.perf_counter()
        response = client.post(URL, json=payload)
        response.raise_for_status()
        times.append((time.perf_counter() - start) * 1000)

results = {
    "total_requests": len(times),
    "successful_requests": len(times),
    "average_response_ms": round(statistics.mean(times), 2),
    "median_response_ms": round(statistics.median(times), 2),
    "fastest_response_ms": round(min(times), 2),
    "slowest_response_ms": round(max(times), 2),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(results, indent=4), encoding="utf-8")

print("\nBenchmark completed successfully!")
print(json.dumps(results, indent=4))
print(f"\nReport saved to: {REPORT_PATH}")
