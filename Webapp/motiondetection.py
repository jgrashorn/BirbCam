import os
import sys
import json
import time
import subprocess
import logging
import fcntl
import atexit
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

def load_env_file():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env_file()

from config import config

# ---------- CONFIG ----------
CAMERAS              = list(config.cameras.keys())
MEDIAMTX_RECORD_DIR  = config.media_dir
CAMERA_DIR           = lambda cam: MEDIAMTX_RECORD_DIR / cam
OUTPUT_DIR           = lambda cam: config.output_dir / cam
EVENTS_DIR           = Path(__file__).parent

MERGE_GAP            = config.merge_gap
PRE_BUFFER           = config.pre_buffer
POST_BUFFER          = config.post_buffer
STABLE_MIN_AGE       = config.stable_min_age
STABLE_POLL_SECONDS  = config.stable_poll_seconds
MAX_AGE_MIN          = config.max_age_minutes
EVENT_RETENTION_DAYS = config.event_retention_days
LOG_LEVEL            = config.log_level

# ---------- LOGGING ----------
logger = logging.getLogger("motion")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)

# ---------- STARTUP VALIDATION ----------
for _cam in CAMERAS:
    for _d in [CAMERA_DIR(_cam), OUTPUT_DIR(_cam)]:
        try:
            _d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {_d}")
        except Exception as _e:
            print(f"Cannot create directory {_d}: {_e}")
            sys.exit(1)

# ---------- SINGLE INSTANCE LOCK ----------
LOCK_PATH = Path("/tmp/birbcam-motion.lock")
_lock_fh = None

def _acquire_single_instance_or_exit():
    global _lock_fh
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fh.write(str(os.getpid()))
        _lock_fh.flush()
    except BlockingIOError:
        print("[motion] another instance is running; exiting")
        sys.exit(0)

def _release_lock():
    try:
        if _lock_fh:
            fcntl.flock(_lock_fh, fcntl.LOCK_UN)
            _lock_fh.close()
    except Exception:
        pass

# ---------- EVENTS JSON (cross-process safe via fcntl) ----------
def _events_path(cam: str) -> Path:
    return EVENTS_DIR / f".motion_events_{cam}.json"

def _events_lock_path(cam: str) -> Path:
    return EVENTS_DIR / f".motion_events_{cam}.lock"

@contextmanager
def _events_flock(cam: str):
    lp = _events_lock_path(cam)
    with open(lp, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def _load_events(cam: str) -> list:
    p = _events_path(cam)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []

def _save_events(cam: str, events: list) -> None:
    with open(_events_path(cam), "w") as f:
        json.dump(events, f, indent=2)

def _cleanup_events(cam: str, events: list) -> list:
    cutoff = time.time() - EVENT_RETENTION_DAYS * 86400
    return [e for e in events if e.get("received", 0) >= cutoff]

# ---------- FILE HELPERS ----------
def _file_is_stable(p: Path) -> bool:
    try:
        now = time.time()
        st1 = p.stat()
        if (now - st1.st_mtime) < STABLE_MIN_AGE:
            logger.debug(f"Unstable (young) {p.name} age={(now - st1.st_mtime):.1f}s < {STABLE_MIN_AGE}s")
            return False
        size1 = st1.st_size
        time.sleep(STABLE_POLL_SECONDS)
        size2 = p.stat().st_size
        if size1 != size2:
            logger.debug(f"Unstable (growing) {p.name}: {size1} -> {size2}")
        return size1 == size2
    except FileNotFoundError:
        return False

def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nk=1:nw=1", str(path)
        ], text=True)
        return float(out.strip())
    except Exception:
        return 0.0

def _run_ffmpeg_trim(src: Path, start: float, end: float, dst: Path) -> bool:
    dur = max(0.0, end - start)
    if dur <= 0.05:
        logger.debug(f"[trim] skip tiny clip {dur:.3f}s from {src.name}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-map", "0:v:0?", "-map", "0:a:0?",
           "-c:v", "copy", "-c:a", "copy",
           "-movflags", "+faststart", "-y", str(dst)]
    if subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size > 0:
        logger.info(f"[trim] copy {src.name} [{start:.1f},{end:.1f}] -> {dst.name}")
        return True
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-map", "0:v:0?", "-map", "0:a:0?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
           "-movflags", "+faststart", "-shortest", "-y", str(dst)]
    ok = subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size > 0
    if ok:
        logger.info(f"[trim] encode {src.name} [{start:.1f},{end:.1f}] -> {dst.name}")
    else:
        logger.warning(f"[trim] FAILED {src.name} [{start:.1f},{end:.1f}]")
    return ok

def _ffmpeg_concat(parts: List[Path], dst: Path) -> bool:
    lst = dst.with_suffix(".txt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with lst.open("w") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(lst),
           "-c", "copy", "-movflags", "+faststart", "-y", str(dst)]
    rc = subprocess.call(cmd)
    lst.unlink(missing_ok=True)
    if rc == 0 and dst.exists() and dst.stat().st_size > 0:
        logger.info(f"[concat] copy {len(parts)} parts -> {dst.name}")
        return True
    with lst.open("w") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(lst),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", "-y", str(dst)]
    rc = subprocess.call(cmd)
    lst.unlink(missing_ok=True)
    ok = rc == 0 and dst.exists() and dst.stat().st_size > 0
    if ok:
        logger.info(f"[concat] encode {len(parts)} parts -> {dst.name}")
    else:
        logger.warning(f"[concat] FAILED {len(parts)} parts -> {dst.name}")
    return ok

# ---------- MAIN PASS ----------
def process_once():
    t0 = time.time()
    logger.info("[run] === STARTED ===")

    for cam in CAMERAS:
        cam_config = config.cameras.get(cam, {})
        if not cam_config.get("motion_detection_enabled", True):
            logger.info(f"[{cam}] motion detection disabled, skipping")
            continue

        logger.info(f"[{cam}] ===== processing =====")

        # Load unprocessed, unclaimed events
        with _events_flock(cam):
            events = _load_events(cam)

        pending = [
            (i, e) for i, e in enumerate(events)
            if not e.get("processed") and not e.get("claimed")
        ]

        if not pending:
            logger.info(f"[{cam}] no pending events")
            continue

        pending.sort(key=lambda x: x[1]["start"])
        logger.info(f"[{cam}] {len(pending)} pending event(s)")

        # Merge nearby events into groups (wall-clock time)
        groups: List[Tuple[float, float, List[int]]] = []
        for idx, e in pending:
            s, en = e["start"], e["end"]
            if groups and s <= groups[-1][1] + MERGE_GAP:
                gs, ge, idxs = groups[-1]
                groups[-1] = (min(gs, s), max(ge, en), idxs + [idx])
            else:
                groups.append((s, en, [idx]))

        logger.info(f"[{cam}] merged into {len(groups)} group(s)")

        # Build file catalog once for all groups
        cam_dir = CAMERA_DIR(cam)
        if not cam_dir.exists():
            logger.warning(f"[{cam}] camera dir missing: {cam_dir}")
            continue

        file_catalog = []
        now = time.time()
        for mp4 in sorted(cam_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime):
            if MAX_AGE_MIN is not None:
                if (now - mp4.stat().st_mtime) > MAX_AGE_MIN * 60:
                    continue
            dur = ffprobe_duration(mp4)
            if dur <= 0:
                continue
            f_end   = mp4.stat().st_mtime
            f_start = f_end - dur
            file_catalog.append((mp4, f_start, dur))

        logger.info(f"[{cam}] catalogued {len(file_catalog)} source file(s)")

        for merged_start, merged_end, idxs in groups:
            clip_start = merged_start - PRE_BUFFER
            clip_end   = merged_end   + POST_BUFFER

            candidates = [
                (mp4, f_start, dur)
                for mp4, f_start, dur in file_catalog
                if clip_start < (f_start + dur) and clip_end > f_start
            ]

            if not candidates:
                logger.info(f"[{cam}] no source files for group at "
                            f"{time.strftime('%H:%M:%S', time.localtime(merged_start))}"
                            f" — marking as expired")
                with _events_flock(cam):
                    events = _load_events(cam)
                    for i in idxs:
                        if i < len(events):
                            events[i]["processed"] = True
                            events[i]["clip"] = None
                    _save_events(cam, events)
                continue

            # Skip group if any needed file isn't stable yet — retry next run
            unstable = [mp4.name for mp4, _, _ in candidates if not _file_is_stable(mp4)]
            if unstable:
                logger.info(f"[{cam}] files not yet stable: {', '.join(unstable)} — will retry next run")
                continue

            # Claim events before processing
            with _events_flock(cam):
                events = _load_events(cam)
                for i in idxs:
                    if i < len(events):
                        events[i]["claimed"] = True
                _save_events(cam, events)

            stamp    = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(merged_start))
            day      = time.strftime("%Y-%m-%d",       time.localtime(merged_start))
            out_dir  = OUTPUT_DIR(cam) / day
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{cam}_{stamp}_motion.mp4"

            if out_path.exists():
                logger.info(f"[{cam}] {out_path.name} already exists, skipping extraction")
                success = True
            elif len(candidates) == 1:
                mp4, f_start, dur = candidates[0]
                ts = max(0.0, clip_start - f_start)
                te = min(dur,  clip_end   - f_start)
                success = _run_ffmpeg_trim(mp4, ts, te, out_path)
            else:
                temp_parts = []
                all_ok = True
                for i, (mp4, f_start, dur) in enumerate(candidates):
                    ts  = max(0.0, clip_start - f_start)
                    te  = min(dur,  clip_end   - f_start)
                    tmp = out_path.parent / f"._tmp_{stamp}_{i}.mp4"
                    if not _run_ffmpeg_trim(mp4, ts, te, tmp):
                        all_ok = False
                        break
                    temp_parts.append(tmp)
                success = _ffmpeg_concat(temp_parts, out_path) if all_ok else False
                for tmp in temp_parts:
                    tmp.unlink(missing_ok=True)

            if success:
                logger.info(f"[{cam}] saved: {out_path.name} (merged {len(idxs)} event(s))")

            # Mark processed and prune old events
            with _events_flock(cam):
                events = _load_events(cam)
                for i in idxs:
                    if i < len(events):
                        events[i]["processed"] = True
                        events[i]["clip"] = str(out_path) if success else None
                events = _cleanup_events(cam, events)
                _save_events(cam, events)

    logger.info(f"[run] === DONE in {time.time() - t0:.2f}s ===")


if __name__ == "__main__":
    _acquire_single_instance_or_exit()
    atexit.register(_release_lock)
    try:
        process_once()
    except KeyboardInterrupt:
        sys.exit(130)