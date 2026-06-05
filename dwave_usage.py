"""
Report D-Wave Leap QPU usage in detail.

Set token first:
    export DWAVE_API_TOKEN='DEV-...'

Run:
    python dwave_usage.py                # all completed jobs
    python dwave_usage.py --status any   # include failed/cancelled
    python dwave_usage.py --dump-json    # also write per-job timing to usage.json

For each job this prints every timing field returned by SAPI, then totals
the quota-relevant ones (qpu_access_time for QPU solvers, charge_time for
hybrid solvers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import requests

US_PER_SEC = 1_000_000
DEFAULT_HOME = "https://na-west-1.cloud.dwavesys.com/sapi/v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--status", default="COMPLETED",
                    help="COMPLETED (default), FAILED, CANCELLED, IN_PROGRESS, PENDING, or 'any'")
    ap.add_argument("--home", default=os.environ.get("SAPI_HOME", DEFAULT_HOME))
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--dump-json", action="store_true", help="Write usage.json with per-job timing")
    args = ap.parse_args()

    token = os.environ.get("DWAVE_API_TOKEN") or os.environ.get("SAPI_TOKEN")
    if not token:
        print("ERROR: set DWAVE_API_TOKEN in the environment.", file=sys.stderr)
        return 2

    home = args.home.rstrip("/")
    s = requests.Session()
    s.headers.update({
        "X-Auth-Token": token,
        "Accept": "application/vnd.dwave.sapi.problems+json; version=3",
    })

    params = {"max_results": args.max}
    if args.status and args.status.lower() != "any":
        params["status"] = args.status

    print(f"GET {home}/problems/  params={params}")
    t0 = time.time()
    r = s.get(f"{home}/problems/", params=params, timeout=args.timeout)
    print(f"  -> HTTP {r.status_code} in {time.time()-t0:.2f}s")
    r.raise_for_status()
    listing = r.json()
    print(f"Got {len(listing)} problems\n")

    if not listing:
        print("No problems returned. Try --status any.")
        return 0

    problem_accept = "application/vnd.dwave.sapi.problem+json; version=3"
    by_solver: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "qpu_access_time": 0,
        "charge_time": 0,
        "run_time": 0,
        "jobs": [],
    })

    # SAPI bins solvers by category; show the category for each
    info_accept = "application/vnd.dwave.sapi.solver-definition+json; version=3"
    solver_kinds: dict[str, str] = {}

    for i, entry in enumerate(listing, 1):
        pid = entry["id"]
        solver = (entry.get("solver") or {}).get("name", "?")
        sub = entry.get("submitted_on", "?")
        status = entry.get("status", "?")

        # Lookup solver category once per solver
        if solver not in solver_kinds:
            try:
                sr = s.get(f"{home}/solvers/remote/{solver}/", timeout=args.timeout,
                           headers={"Accept": info_accept})
                if sr.status_code == 200:
                    body = sr.json()
                    cat = body.get("category") or body.get("properties", {}).get("category", "?")
                    qcr = body.get("properties", {}).get("quota_conversion_rate", "?")
                    solver_kinds[solver] = f"{cat} (quota_rate={qcr})"
                else:
                    solver_kinds[solver] = "?"
            except Exception:
                solver_kinds[solver] = "?"

        t0 = time.time()
        ar = s.get(f"{home}/problems/{pid}", timeout=args.timeout,
                   headers={"Accept": problem_accept})
        elapsed = time.time() - t0

        print(f"[{i}/{len(listing)}] {pid}")
        print(f"    submitted: {sub}")
        print(f"    solver:    {solver}  [{solver_kinds[solver]}]")
        print(f"    status:    {status}   (HTTP {ar.status_code}, {elapsed:.2f}s)")

        if ar.status_code != 200:
            print(f"    -- skipped --\n")
            continue

        body = ar.json()
        answer = body.get("answer") or {}
        timing = answer.get("timing") or {}

        # Print every timing field returned
        print(f"    timing fields:")
        if not timing:
            print(f"      (empty)")
        for k, v in sorted(timing.items()):
            if isinstance(v, (int, float)) and "time" in k:
                print(f"      {k:<32} {v:>15,} us  ({v/US_PER_SEC:.6f} s)")
            else:
                print(f"      {k:<32} {v}")

        # Accumulate
        bucket = by_solver[solver]
        bucket["count"] += 1
        for field in ("qpu_access_time", "charge_time", "run_time"):
            v = timing.get(field) or 0
            try:
                bucket[field] += int(v)
            except (TypeError, ValueError):
                pass
        bucket["jobs"].append({
            "id": pid, "submitted_on": sub, "status": status,
            "timing": timing,
        })
        print()

    # ---- summary ----
    print("=" * 78)
    print("PER-SOLVER TOTALS")
    print("=" * 78)
    grand = {"qpu_access_time": 0, "charge_time": 0, "run_time": 0, "count": 0}
    for name, b in by_solver.items():
        grand["count"] += b["count"]
        for k in ("qpu_access_time", "charge_time", "run_time"):
            grand[k] += b[k]
        print(f"\n{name}  [{solver_kinds.get(name, '?')}]")
        print(f"  jobs:             {b['count']}")
        print(f"  qpu_access_time:  {b['qpu_access_time']:>15,} us  ({b['qpu_access_time']/US_PER_SEC:.6f} s)")
        print(f"  charge_time:      {b['charge_time']:>15,} us  ({b['charge_time']/US_PER_SEC:.6f} s)")
        print(f"  run_time:         {b['run_time']:>15,} us  ({b['run_time']/US_PER_SEC:.6f} s)")

    print()
    print("=" * 78)
    print("GRAND TOTALS")
    print("=" * 78)
    print(f"  total jobs:       {grand['count']}")
    print(f"  qpu_access_time:  {grand['qpu_access_time']:>15,} us  "
          f"= {grand['qpu_access_time']/US_PER_SEC:.6f} s "
          f"= {grand['qpu_access_time']/US_PER_SEC/60:.6f} min")
    print(f"  charge_time:      {grand['charge_time']:>15,} us  "
          f"= {grand['charge_time']/US_PER_SEC:.6f} s "
          f"= {grand['charge_time']/US_PER_SEC/60:.6f} min")
    print(f"  run_time:         {grand['run_time']:>15,} us  "
          f"= {grand['run_time']/US_PER_SEC:.6f} s "
          f"= {grand['run_time']/US_PER_SEC/60:.6f} min")
    print()
    print("Notes on what counts against quota:")
    print("  - QPU solvers (Advantage_*):  qpu_access_time is the actual QPU time used.")
    print("  - Hybrid solvers (hybrid_*):  charge_time is what bills against the hybrid quota.")
    print("  - quota_conversion_rate above shows how each solver's run_time maps to quota seconds.")
    print("  - Developer (free) plan: 60 s QPU + 1200 s hybrid solver per month.")
    print("  - SAPI retains problem history ~30 days.")

    if args.dump_json:
        out = {"by_solver": {k: v for k, v in by_solver.items()}, "totals": grand}
        with open("usage.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("\nWrote usage.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
