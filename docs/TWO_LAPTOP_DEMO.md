# Two-laptop deployment

Laptop 1 hosts the frozen model, graph/data, HMM/Viterbi and FastAPI. Laptop 2 runs only the evaluator UI. Lakshya operates the UI; Nakul hosts the API, with Nikhil supporting verification. Screenshots and deployment claims must identify the machines actually used, not imply an unobserved network test.

## Laptop 1 model and API

Activate the project Python environment, install `requirements.txt`, and place organiser evaluator files directly under `data/evaluator_test/TRIP_ID/TRIP_ID.{track,nodes,arcs,route}`. Raw data is excluded from Git; a clone does not include it. Final model/config files are included, so retraining is unnecessary.

```bash
python -m pathsnap.api.app --host 0.0.0.0 --port 8003 --trips-dir data/evaluator_test
```

For development select `data/student_test`. Keep the server running. If the port is occupied, identify the existing process; reuse a healthy correct server or stop that verified old API before restarting. Do not repeatedly start extra copies. The application default is 8000; all examples here explicitly choose 8003.

Open `http://localhost:8003/health` on Laptop 1. Expected: `model_loaded: true`, `release_id: pathsnap-monotone-v1`, `trips_dir: evaluator_test`, `n_trips: 10`. Hashes must match `artifacts/final_release.json`. Restart after any intentional artifact change; a browser refresh does not reload Python's model cache.

Find the current active Wi-Fi IPv4: Windows `ipconfig` (ignore VMware/VPN adapters); macOS Wi-Fi details or `ipconfig getifaddr en0` where en0 is the active Wi-Fi interface. Recheck the address after switching networks.

## Laptop 2 UI

Only Python's standard library and the `ui/` folder are needed. No model, ML dependencies or raw data are required.

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory ui
```

Open `http://LAPTOP_1_IP:8003/health` from Laptop 2 first. Once it returns the expected JSON, open `http://localhost:5500`, enter `http://LAPTOP_1_IP:8003` in the API field, click Connect, select a trip and keep Run fresh inference checked. The UI also accepts `?api=http://LAPTOP_1_IP:8003` for prefilling.

Click Predict route. Confirm a successful POST appears in Laptop 1 logs from Laptop 2's address. Capture the receiving terminal and UI result. Localhost tests alone do not establish the physical two-laptop flow. All ten reserved trips must be selectable, not a hardcoded example.

```bash
python scripts/smoke_test_api.py --api http://LAPTOP_1_IP:8003
```

This optional standard-library test exercises all configured trips. Do not use reserved scores to change the trained model, features, thresholds or decoder.

## Network troubleshooting

- Both devices need a reachable local network. Campus Wi-Fi may isolate devices; a shared phone hotspot is an alternative. Check the actual SSID on both devices and prevent unwanted automatic switching.
- Localhost health works but remote health times out: check the current IP, server binding, Windows inbound rules and client isolation before changing code.
- On a trusted personal hotspot, Windows can use a Private profile. Keep the campus network Public. A specific active Block rule can override an Allow rule; inspect the Python executable/profile before changing only the relevant rule. Keep the firewall enabled.
- If an administrator creates a temporary inbound TCP 8003 rule for the demo, remove it afterward. Do not disable unrelated rules or the whole firewall.
- Frozen checksum mismatch on Windows: LF checkout is enforced by `.gitattributes`. Compare against the manifest; restore the exact committed config bytes if needed. Do not change the manifest to hide a mismatch.
- UI still points to 127.0.0.1: replace its API field with Laptop 1's actual IP and chosen port. The query parameter avoids stale stored addresses.
- Basemap tiles need internet; Leaflet and road/GPS geometry are bundled/returned locally.

## Evidence status

Retained automatic checks and report screenshots are local HTTP/development evidence. A team-reported network success is distinct from a captured receiving-terminal screenshot. Include actual evidence of the final physical demonstration; never relabel local captures as remote ones.
