"""Check a running API end to end, from any laptop on the network.

    python scripts/smoke_test_api.py --api http://192.168.1.20:8000

Calls /health, /trips, then /predict for EVERY listed trip and prints a table.
Use it on Laptop 2 before the demo to prove all ten trips work.
"""
import argparse
import json
import time
import urllib.request


def get(url, data=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://127.0.0.1:8000")
a = ap.parse_args()
api = a.api.rstrip("/")
h = get(api + "/health")
print("health:", h["status"], "| model loaded:", h["model_loaded"], "| trips:", h["n_trips"])
trips = get(api + "/trips")["trips"]
ok = 0
print(f"{'trip':<10}{'points':>8}{'secs':>7}{'HMM acc':>9}{'CLF acc':>9}{'routeF1':>9}{'cont':>7}")
for t in trips:
    t0 = time.time()
    try:
        d = get(api + "/predict", {"trip_id": t["trip_id"]})
        m = d["metrics"]
        h, c = m["hmm"] or {}, m["classifier_only"] or {}
        print(f"{t['trip_id']:<10}{d['n_observations']:>8}{time.time() - t0:>7.1f}{h.get('point_accuracy', float('nan')):>9.3f}"
              f"{c.get('point_accuracy', float('nan')):>9.3f}{h.get('route_f1', float('nan')):>9.3f}"
              f"{h.get('route_continuity', float('nan')):>7.3f}")
        ok += 1
    except Exception as e:  # noqa: BLE001
        print(f"{t['trip_id']:<10} FAILED: {e}")
print(f"\n{ok}/{len(trips)} trips predicted successfully")
