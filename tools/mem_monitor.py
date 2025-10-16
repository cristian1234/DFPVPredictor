# python
# File: tools/mem_monitor.py
# Lightweight memory monitor using tracemalloc + psutil

import os
import time
import threading
import tracemalloc
import psutil

SNAP_DIR = os.path.join("tools", "mem_snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

_periodic_thread = None
_stop_event = None
_prev_snapshot = None

def start(period=10):
    """
    Start tracemalloc and a background thread that logs RSS + traced memory every `period` seconds.
    Call once at program start.
    """
    global _periodic_thread, _stop_event
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)  # keep 25 frames
    if _periodic_thread is not None:
        return
    _stop_event = threading.Event()
    def _periodic():
        proc = psutil.Process()
        while not _stop_event.is_set():
            rss = proc.memory_info().rss
            traced_current, traced_peak = tracemalloc.get_traced_memory()
            print(f"[mem-monitor] RSS={rss:,} bytes | traced_current={traced_current:,} | traced_peak={traced_peak:,}")
            _stop_event.wait(period)
    _periodic_thread = threading.Thread(target=_periodic, daemon=True)
    _periodic_thread.start()

def snapshot(tag):
    """
    Take a tracemalloc snapshot and print top 20 allocs diff vs previous snapshot (if exists).
    Saves snapshot text to tools/mem_snapshots/{tag}.txt.
    Use tags like 'before_train', 'after_train', 'after_test'.
    """
    global _prev_snapshot
    snap = tracemalloc.take_snapshot()
    out_path = os.path.join(SNAP_DIR, f"{int(time.time())}_{tag}.txt")
    with open(out_path, "w") as f:
        f.write(f"Snapshot: {tag}\n")
        f.write(f"Time: {time.ctime()}\n\n")
        # If we have a previous snapshot, compute differences
        if _prev_snapshot is not None:
            stats = snap.compare_to(_prev_snapshot, "lineno")
            f.write("Top 20 differences (current vs previous):\n")
            for stat in stats[:20]:
                f.write(str(stat) + "\n")
            print(f"[mem-monitor] Wrote diff snapshot {out_path}")
        else:
            stats = snap.statistics("lineno")
            f.write("Top 20 current allocations:\n")
            for stat in stats[:20]:
                f.write(str(stat) + "\n")
            print(f"[mem-monitor] Wrote initial snapshot {out_path}")
    _prev_snapshot = snap

def stop():
    """
    Stop the background thread and tracemalloc.
    Call at program exit.
    """
    global _stop_event, _periodic_thread
    if _stop_event is not None:
        _stop_event.set()
    if _periodic_thread is not None:
        _periodic_thread.join(timeout=1.0)
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    print("[mem-monitor] stopped")
