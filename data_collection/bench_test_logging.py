"""
Standalone throughput/packet-loss bench test for the LogConfig blocks in config.py.

Connects and starts all 8 LogConfig blocks WITHOUT arming or flying, runs for a fixed
duration, and reports measured vs requested packet rate per block plus any error_cb firings.
This is the risk called out in README.md/elaborate_plan.md: ~370 combined log packets/sec
across 8 blocks (2 at 100 Hz), concurrent with commander setpoints during real flight, is
untested on this specific radio link -- run this BEFORE ever arming the Crazyflie.

If measured rates are well below requested, or error_cb fires, reduce rates in config.py
(drop the `baro` block first, then halve `imu`/`flow` from 100->50 Hz) before flying.

Run from inside this directory:
    source ~/.venv/bin/activate
    cd data_collection
    python bench_test_logging.py
"""

from __future__ import annotations

import time

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import cf_link
import config

BENCH_DURATION_S = 30.0


def main() -> None:
    cf_link.initialize_cflib_drivers()

    counts = {block["name"]: 0 for block in config.LOG_BLOCKS}
    errors = {block["name"]: 0 for block in config.LOG_BLOCKS}

    def make_data_cb(name):
        def cb(timestamp_ms, data, logconf):
            counts[name] += 1
        return cb

    def make_error_cb(name):
        def cb(logconf, msg):
            errors[name] += 1
            print(f"error on block '{name}': {msg}")
        return cb

    print(f"Connecting to {config.URI} (props off, NOT arming)...")
    with SyncCrazyflie(config.URI) as scf:
        cf = scf.cf
        log_configs = []
        for block in config.LOG_BLOCKS:
            lg = LogConfig(name=block["name"], period_in_ms=block["period_ms"])
            for cf_name, cf_type in block["vars"]:
                lg.add_variable(cf_name, cf_type)
            lg.data_received_cb.add_callback(make_data_cb(block["name"]))
            lg.error_cb.add_callback(make_error_cb(block["name"]))
            cf.log.add_config(lg)
            lg.start()
            log_configs.append(lg)

        print(f"Running for {BENCH_DURATION_S:.0f}s... (do not arm/fly during this test)")
        time.sleep(BENCH_DURATION_S)

        for lg in log_configs:
            lg.stop()

    print(f"\n{'block':10s}  {'requested_hz':>12s}  {'measured_hz':>11s}  errors")
    any_warning = False
    for block in config.LOG_BLOCKS:
        name = block["name"]
        requested_hz = 1000.0 / block["period_ms"]
        measured_hz = counts[name] / BENCH_DURATION_S
        print(f"{name:10s}  {requested_hz:12.1f}  {measured_hz:11.1f}  {errors[name]}")
        if measured_hz < 0.8 * requested_hz:
            any_warning = True

    total_pkts_per_s = sum(counts.values()) / BENCH_DURATION_S
    print(f"\nTotal measured packet rate: {total_pkts_per_s:.1f} pkt/s")

    if any(errors.values()) or any_warning:
        print(
            "\nWARNING: measured rate well below requested and/or error_cb fired for at "
            "least one block. Reduce log rates in config.py before flying -- drop 'baro' "
            "first, then halve 'imu'/'flow' from 100 -> 50 Hz."
        )
    else:
        print("\nAll blocks kept up with their requested rate; no errors reported.")


if __name__ == "__main__":
    main()
