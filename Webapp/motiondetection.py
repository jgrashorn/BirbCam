import os
import sys
import json
import time
import shutil
import subprocess
import logging
import fcntl, atexit
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np

# ---------- CONFIG ----------
CAMERA = os.environ.get("BIRBCAM_CAMERA", "garten")

# Root directory where MediaMTX recorder writes MP4s
MEDIAMTX_RECORD_DIR = Path(os.environ.get("BIRBCAM_MEDIA_DIR", "/home/birb/mediamtx/recordings"))

# Only scan the camera subfolder if it exists; else fallback to full tree search
CAMERA_DIR = (MEDIAMTX_RECORD_DIR / CAMERA)

# Output folder inside your Webapp
OUTPUT_DIR = Path(os.environ.get("BIRBCAM_OUTPUT_DIR", f"/home/birb/BirbCam/Webapp/recordings/{CAMERA}"))
STATE_FILE = OUTPUT_DIR.parent / f".motion_state_{CAMERA}.json"

# Motion detection parameters
SAMPLE_FPS = float(os.environ.get("BIRBCAM_SAMPLE_FPS", "2"))  # decode ~N fps
RESIZE_W = int(os.environ.get("BIRBCAM_RESIZE_W", "180"))
DIFF_THRESH = int(os.environ.get("BIRBCAM_DIFF_THRESH", "25"))
MIN_CHANGED_PIXELS = int(os.environ.get("BIRBCAM_MIN_CHANGED_PIXELS", "200"))
MIN_MOTION_FRAMES = int(os.environ.get("BIRBCAM_MIN_MOTION_FRAMES", "1"))
MERGE_GAP = float(os.environ.get("BIRBCAM_MERGE_GAP", "1.5"))  # seconds

# Export behavior
PRE_BUFFER = float(os.environ.get("BIRBCAM_PRE_BUFFER", "2.0"))
POST_BUFFER = float(os.environ.get("BIRBCAM_POST_BUFFER", "3.0"))
TAIL_NEAR_END = float(os.environ.get("BIRBCAM_TAIL_NEAR_END", "1.0"))
HEAD_NEAR_START = float(os.environ.get("BIRBCAM_HEAD_NEAR_START", "1.0"))

# Limit scanning to recent files to save time (set None to disable)
MAX_AGE_MIN = int(os.environ.get("BIRBCAM_MAX_AGE_MIN", "360"))  # 6h

# Consider a file "stable" if it's older than this and its size doesn't change briefly
STABLE_MIN_AGE = float(os.environ.get("BIRBCAM_STABLE_MIN_AGE", "5.0"))  # seconds
STABLE_POLL_SECONDS = float(os.environ.get("BIRBCAM_STABLE_POLL_SECONDS", "1.0"))

# Debug/control
LOG_LEVEL = os.environ.get("BIRBCAM_LOG", "DEBUG").upper()
MAX_FILES = int(os.environ.get("BIRBCAM_MAX_FILES", "20"))  # cap files processed per run
DRY_RUN = os.environ.get("BIRBCAM_DRY_RUN", "0") == "1"

logger = logging.getLogger("motion")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)

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

# ---------- HELPERS ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_state(state: Dict):
    ensure_dir(OUTPUT_DIR)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))

def _prune_state(processed: set, produced: set, existing_inputs: set) -> tuple[set, set]:
    # Keep only processed input files that still exist
    pruned_processed = {p for p in processed if p in existing_inputs and Path(p).exists()}
    # Keep only produced outputs that still exist in OUTPUT_DIR
    pruned_produced = {n for n in produced if (OUTPUT_DIR / n).exists()}
    return pruned_processed, pruned_produced

def list_camera_files() -> List[Path]:
    if CAMERA_DIR.exists():
        base = CAMERA_DIR
        logger.debug(f"Scanning camera dir: {base}")
        files = sorted(base.rglob("*.mp4"), key=lambda p: (p.stat().st_mtime, str(p)))
    else:
        base = MEDIAMTX_RECORD_DIR
        logger.debug(f"Scanning root dir: {base} (filtering by camera={CAMERA})")
        files = []
        if base.exists():
            for p in base.rglob("*.mp4"):
                if CAMERA in p.parts:
                    files.append(p)
            files.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    if MAX_AGE_MIN is not None:
        now = time.time()
        before = len(files)
        files = [p for p in files if (now - p.stat().st_mtime) <= MAX_AGE_MIN * 60]
        logger.debug(f"Filtered by age: {before} -> {len(files)} (<= {MAX_AGE_MIN} min)")
    return files

def _file_is_stable(p: Path) -> bool:
    try:
        now = time.time()
        st1 = p.stat()
        if (now - st1.st_mtime) < STABLE_MIN_AGE:
            logger.debug(f"Unstable (young) {p} age={(now - st1.st_mtime):.2f}s < {STABLE_MIN_AGE}s")
            return False
        size1 = st1.st_size
        time.sleep(STABLE_POLL_SECONDS)
        size2 = p.stat().st_size
        stable = size1 == size2
        if not stable:
            logger.debug(f"Unstable (growing) {p} size {size1} -> {size2}")
        return stable
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

def detect_motion_windows(video_path: Path) -> Tuple[List[Tuple[float, float]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"OpenCV failed to open {video_path}")
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = ffprobe_duration(video_path)
    if not duration or duration <= 0:
        duration = (total_frames / fps) if fps and fps > 0 else 0.0

    step = max(int(round(fps / max(SAMPLE_FPS, 0.1))), 1)
    logger.debug(f"[scan] {video_path.name}: fps={fps:.2f} frames={total_frames} dur={duration:.3f}s step={step}")

    # Watchdog: don't let decoding of a single file run forever
    start_wall = time.time()
    hard_limit = max(30.0, duration + 5.0)  # at least 30s, or duration+5s

    prev = None
    motion_open = False
    motion_start = 0.0
    motion_frames = 0
    windows: List[Tuple[float, float]] = []

    frame_idx = 0
    processed_frames = 0

    while True:
        if (time.time() - start_wall) > hard_limit:
            logger.warning(f"[scan] timeout on {video_path.name} after {time.time() - start_wall:.1f}s")
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % step != 0:
            continue
        processed_frames += 1
        if processed_frames % 100 == 0:
            logger.debug(f"[scan] {video_path.name}: processed ~{processed_frames} frames")

        h, w = frame.shape[:2]
        scale = RESIZE_W / max(w, 1)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is None:
            prev = gray
            continue

        diff = cv2.absdiff(prev, gray)
        _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        changed = int(np.count_nonzero(mask))
        t = (frame_idx - 1) / max(fps, 1.0)

        if changed >= MIN_CHANGED_PIXELS:
            motion_frames += 1
            if not motion_open and motion_frames >= MIN_MOTION_FRAMES:
                motion_open = True
                motion_start = t
        else:
            motion_frames = max(0, motion_frames - 1)
            if motion_open and motion_frames == 0:
                motion_open = False
                windows.append((motion_start, t))
        prev = gray

    if motion_open:
        windows.append((motion_start, duration))

    cap.release()

    # Merge windows that are close
    merged: List[Tuple[float, float]] = []
    for s, e in sorted(windows):
        if not merged:
            merged.append((s, e))
            continue
        ls, le = merged[-1]
        if s <= le + MERGE_GAP:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))

    # Apply pre/post buffers clipped to duration
    buffered: List[Tuple[float, float]] = []
    for s, e in merged:
        bs = max(0.0, s - PRE_BUFFER)
        be = min(duration, e + POST_BUFFER)
        buffered.append((bs, be))

    logger.info(f"[windows] {video_path.name}: {len(buffered)} window(s) -> {buffered}")
    return buffered, duration

def _run_ffmpeg_trim(src: Path, start: float, end: float, dst: Path, reencode_fallback=True) -> bool:
    dur = max(0.0, end - start)
    if dur <= 0.05:
        logger.debug(f"[trim] skip tiny clip {dur:.3f}s from {src.name}")
        return False
    if DRY_RUN:
        logger.info(f"[trim] DRY {src.name} [{start:.3f},{end:.3f}] -> {dst.name}")
        return True
    # Try stream copy first (include audio if present)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(src),
        "-map", "0:v:0?", "-map", "0:a:0?",      # include audio optionally
        "-c:v", "copy",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-y", str(dst)
    ]
    rc = subprocess.call(cmd)
    if rc == 0 and dst.exists() and dst.stat().st_size > 0:
        return True
    if not reencode_fallback:
        return False
    # Fallback re-encode if copy failed due to keyframe alignment (keep audio)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(src),
        "-map", "0:v:0?", "-map", "0:a:0?",      # include audio optionally
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",                              # trim to shortest stream
        "-y", str(dst)
    ]
    rc = subprocess.call(cmd)
    return rc == 0 and dst.exists() and dst.stat().st_size > 0

def _ffmpeg_concat(filelist: List[Path], dst: Path) -> bool:
    if DRY_RUN:
        logger.info(f"[concat] DRY {len(filelist)} chunks -> {dst.name}")
        return True
    tmp_list = dst.with_suffix(".txt")
    with tmp_list.open("w") as f:
        for p in filelist:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(tmp_list),
        "-c", "copy",
        "-movflags", "+faststart",
        "-y", str(dst)
    ]
    rc = subprocess.call(cmd)
    try:
        tmp_list.unlink(missing_ok=True)
    except Exception:
        pass
    if rc != 0 or not dst.exists() or dst.stat().st_size == 0:
        # Fallback re-encode
        with tmp_list.open("w") as f:
            for p in filelist:
                f.write(f"file '{p.as_posix()}'\n")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(tmp_list),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-y", str(dst)
        ]
        rc = subprocess.call(cmd)
        try:
            tmp_list.unlink(missing_ok=True)
        except Exception:
            pass
    return rc == 0 and dst.exists() and dst.stat().st_size > 0

def sanitize_name(p: Path) -> str:
    parts = list(p.parts)
    tokens = parts[-3:-1] + [p.stem]
    return "_".join(tokens)

# ---------- MAIN PASS ----------

def process_once():
    t0 = time.time()
    ensure_dir(OUTPUT_DIR)
    state = load_state()
    processed = set(state.get("processed_files", []))
    produced = set(state.get("produced_outputs", []))

    files = list_camera_files()
    logger.info(f"[start] found {len(files)} file(s) for camera={CAMERA}")

    # Prune state to existing files/outputs right away
    existing_inputs = set(str(p) for p in files)
    before_proc, before_prod = len(processed), len(produced)
    processed, produced = _prune_state(processed, produced, existing_inputs)
    if before_proc != len(processed) or before_prod != len(produced):
        logger.debug(f"[state] pruned processed {before_proc}->{len(processed)}, produced {before_prod}->{len(produced)}")
        state["processed_files"] = sorted(processed)
        state["produced_outputs"] = sorted(produced)
        save_state(state)

    # Skip files already processed
    files_to_scan = [p for p in files if str(p) not in processed]
    if not files_to_scan:
        logger.info("[start] nothing new to process")
        return

    # Cap per run for CPU control
    files_to_scan = files_to_scan[-MAX_FILES:]
    logger.info(f"[start] scanning up to {len(files_to_scan)} new file(s) (MAX_FILES={MAX_FILES})")

    # Pre-compute windows and durations
    file_info: Dict[Path, Dict] = {}
    for p in files_to_scan:
        if not _file_is_stable(p):
            logger.debug(f"[skip] unstable: {p}")
            continue
        try:
            windows, dur = detect_motion_windows(p)
            file_info[p] = {"windows": windows, "duration": dur}
        except Exception as e:
            logger.exception(f"[error] scanning {p}: {e}")

    if not file_info:
        logger.info("[scan] no stable files with data")
        return

    # DO NOT reset processed here
    # processed = set([])  # REMOVE this line if present

    # Iterate and export
    keys = list(file_info.keys())
    for idx, p in enumerate(keys):
        info = file_info[p]
        windows = info["windows"]
        duration = info["duration"]
        logger.info(f"[export] {p.name}: {len(windows)} window(s)")

        next_p = keys[idx + 1] if idx + 1 < len(keys) else None
        next_info = file_info.get(next_p) if next_p else None
        next_windows = list(next_info["windows"]) if next_info else []

        consumed_next = set()

        for wi, (s, e) in enumerate(windows):
            tail_touch = (duration - e) <= TAIL_NEAR_END
            head_touch_idx = None
            if tail_touch and next_p and next_windows:
                for nwi, (ns, ne) in enumerate(next_windows):
                    if ns <= HEAD_NEAR_START:
                        head_touch_idx = nwi
                        break

            if head_touch_idx is not None:
                base = sanitize_name(p)
                stamp = f"{int(time.time())}"
                tmp1 = OUTPUT_DIR / f"._tmp_{base}_{wi}_{stamp}_a.mp4"
                tmp2 = OUTPUT_DIR / f"._tmp_{base}_{wi}_{stamp}_b.mp4"
                out = OUTPUT_DIR / f"{base}_{wi}_{stamp}_joined.mp4"
                logger.debug(f"[join] tail->head across files: {p.name} -> {next_p.name}")

                ok1 = _run_ffmpeg_trim(p, s, duration, tmp1)
                ns, ne = next_windows[head_touch_idx]
                ok2 = _run_ffmpeg_trim(next_p, 0.0, ne, tmp2)
                if ok1 and ok2:
                    ok = _ffmpeg_concat([tmp1, tmp2], out)
                    if ok:
                        logger.info(f"[save] {out.name}")
                        produced.add(out.name)
                        consumed_next.add(head_touch_idx)
                for t in (tmp1, tmp2):
                    try:
                        t.unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                base = sanitize_name(p)
                stamp = f"{int(time.time())}"
                day = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
                day_dir = OUTPUT_DIR / day
                day_dir.mkdir(parents=True, exist_ok=True)
                out = day_dir / f"{base}_{wi}_{stamp}.mp4"
                if out.name in produced:
                    logger.debug(f"[dupe] {out.name} already produced; skipping")
                    continue
                logger.debug(f"[trim] {p.name} [{s:.3f},{e:.3f}] -> {out.name}")
                ok = _run_ffmpeg_trim(p, s, e, out)
                if ok:
                    logger.info(f"[save] {out.name}")
                    produced.add(out.name)

        if next_p and consumed_next:
            for nwi in consumed_next:
                next_windows[nwi] = (-1.0, -1.0)

        processed.add(str(p))
        # Save progress incrementally, but prune first to avoid growth
        processed, produced = _prune_state(processed, produced, existing_inputs)
        state["processed_files"] = sorted(processed)
        state["produced_outputs"] = sorted(produced)
        save_state(state)

    logger.info(f"[done] processed {len(keys)} file(s) in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    _acquire_single_instance_or_exit()
    atexit.register(_release_lock)
    try:
        process_once()
    except KeyboardInterrupt:
        sys.exit(130)
