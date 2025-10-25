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

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load environment before importing config
load_env_file()

import cv2
import numpy as np
from config import config

# ---------- CONFIG ----------
# Use centralized configuration
CAMERA = config.camera
MEDIAMTX_RECORD_DIR = config.media_dir
CAMERA_DIR = (MEDIAMTX_RECORD_DIR / CAMERA)
OUTPUT_DIR = config.output_dir
STATE_FILE = OUTPUT_DIR.parent / f".motion_state_{CAMERA}.json"

# Motion detection parameters
SAMPLE_FPS = config.sample_fps
RESIZE_W = config.resize_width
DIFF_THRESH = config.diff_threshold
MIN_CHANGED_PIXELS = config.min_changed_pixels
MIN_MOTION_FRAMES = config.min_motion_frames
MERGE_GAP = config.merge_gap

# Optimization settings
USE_FFMPEG_PREPROCESSING = config.use_ffmpeg_preprocessing
TEMP_DIR = config.temp_dir

# Export behavior
PRE_BUFFER = config.pre_buffer
POST_BUFFER = config.post_buffer
TAIL_NEAR_END = config.tail_near_end
HEAD_NEAR_START = config.head_near_start

# Processing limits
MAX_AGE_MIN = config.max_age_minutes
STABLE_MIN_AGE = config.stable_min_age
STABLE_POLL_SECONDS = config.stable_poll_seconds

# Debug/control
LOG_LEVEL = config.log_level
MAX_FILES = config.max_files_per_run
DRY_RUN = config.dry_run

logger = logging.getLogger("motion")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)

# Validate configuration at startup
try:
    config.validate_directories(create=True)
    logger.info(f"Configuration validated successfully for camera: {CAMERA}")
    logger.info(f"Processing mode: {'FFmpeg preprocessing' if USE_FFMPEG_PREPROCESSING else 'Direct OpenCV'}")
    logger.info(f"Media directory: {MEDIAMTX_RECORD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    sys.exit(1)

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
    probe_start = time.time()
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nk=1:nw=1", str(path)
        ], text=True)
        duration = float(out.strip())
        probe_time = time.time() - probe_start
        logger.debug(f"[timing] ffprobe {path.name}: {probe_time:.3f}s -> {duration:.2f}s duration")
        return duration
    except Exception as e:
        probe_time = time.time() - probe_start
        logger.debug(f"[timing] ffprobe {path.name}: {probe_time:.3f}s -> FAILED: {e}")
        return 0.0

def create_preprocessed_video(video_path: Path) -> Path:
    """Create a low-resolution, grayscale version of the video for motion detection."""
    preprocess_start = time.time()
    
    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate temp filename
    temp_name = f"motion_{video_path.stem}_{int(time.time())}.mp4"
    temp_path = TEMP_DIR / temp_name
    
    # Calculate target height maintaining aspect ratio
    # We'll let FFmpeg figure out the height automatically
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        
        # Video filters: scale down, convert to grayscale, and apply slight blur
        "-vf", f"scale={RESIZE_W}:-2,format=gray,gblur=sigma=1",
        
        # Sampling: only process every Nth frame based on SAMPLE_FPS
        "-r", str(SAMPLE_FPS),
        
        # Fast encoding settings for temporary file
        "-c:v", "libx264", 
        "-preset", "ultrafast",  # Fastest encoding
        "-crf", "35",            # Higher CRF for smaller file (quality doesn't matter much)
        "-tune", "fastdecode",   # Optimize for fast decoding
        
        # No audio needed for motion detection
        "-an",
        
        # Overwrite output
        "-y", str(temp_path)
    ]
    
    logger.debug(f"[preprocess] Creating low-res version: {video_path.name} -> {temp_path.name}")
    
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    preprocess_time = time.time() - preprocess_start
    
    if rc != 0 or not temp_path.exists() or temp_path.stat().st_size == 0:
        logger.warning(f"[preprocess] FAILED to create preprocessed video for {video_path.name} in {preprocess_time:.3f}s")
        # Clean up failed file
        try:
            temp_path.unlink(missing_ok=True)
        except:
            pass
        return None
    
    original_size = video_path.stat().st_size
    processed_size = temp_path.stat().st_size
    compression_ratio = processed_size / original_size if original_size > 0 else 0
    
    logger.info(f"[timing] ffmpeg preprocess {video_path.name}: {preprocess_time:.3f}s, "
               f"size reduction {original_size//1024//1024}MB -> {processed_size//1024//1024}MB "
               f"({compression_ratio:.1%})")
    
    return temp_path

def cleanup_temp_file(temp_path: Path):
    """Clean up temporary preprocessed video file."""
    try:
        if temp_path and temp_path.exists():
            temp_path.unlink()
            logger.debug(f"[cleanup] Removed temp file {temp_path.name}")
    except Exception as e:
        logger.debug(f"[cleanup] Failed to remove {temp_path}: {e}")

def cleanup_old_temp_files():
    """Clean up old temporary files on startup."""
    if not TEMP_DIR.exists():
        return
        
    cleanup_start = time.time()
    cleaned = 0
    
    try:
        for temp_file in TEMP_DIR.glob("motion_*.mp4"):
            try:
                # Remove files older than 1 hour or very small files (failed processing)
                age = time.time() - temp_file.stat().st_mtime
                size = temp_file.stat().st_size
                
                if age > 3600 or size < 1024:  # 1 hour or < 1KB
                    temp_file.unlink()
                    cleaned += 1
                    
            except Exception as e:
                logger.debug(f"[cleanup] Failed to remove old temp file {temp_file}: {e}")
                
        cleanup_time = time.time() - cleanup_start
        if cleaned > 0:
            logger.debug(f"[cleanup] Removed {cleaned} old temp files in {cleanup_time:.3f}s")
            
    except Exception as e:
        logger.debug(f"[cleanup] Error during temp file cleanup: {e}")

def detect_motion_windows(video_path: Path) -> Tuple[List[Tuple[float, float]], float]:
    motion_start_time = time.time()
    temp_video_path = None

    # Timing accumulators
    total_read_time = 0.0
    total_resize_time = 0.0
    total_convert_time = 0.0
    total_blur_time = 0.0
    total_diff_time = 0.0
    
    # Duration probe (for original video timing)
    duration = ffprobe_duration(video_path)
    
    # Choose processing path based on configuration
    if USE_FFMPEG_PREPROCESSING:
        # Use FFmpeg preprocessing approach
        temp_video_path = create_preprocessed_video(video_path)
        if temp_video_path is None:
            logger.warning(f"[motion] FFmpeg preprocessing failed, falling back to direct OpenCV for {video_path.name}")
            processing_path = video_path
            use_preprocessing = False
        else:
            processing_path = temp_video_path
            use_preprocessing = True
    else:
        # Direct OpenCV approach (original method)
        processing_path = video_path
        use_preprocessing = False
    
    # Time video opening
    open_start = time.time()
    cap = cv2.VideoCapture(str(processing_path))
    if not cap.isOpened():
        logger.warning(f"OpenCV failed to open {processing_path}")
        if temp_video_path:
            cleanup_temp_file(temp_video_path)
        return [], 0.0
    open_time = time.time() - open_start

    # Get video properties from the processing video
    fps = cap.get(cv2.CAP_PROP_FPS) or SAMPLE_FPS if use_preprocessing else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # For preprocessed video, we don't need to calculate duration again
    if not duration or duration <= 0:
        # Fallback if duration probe failed
        original_cap = cv2.VideoCapture(str(video_path))
        if original_cap.isOpened():
            original_fps = original_cap.get(cv2.CAP_PROP_FPS) or 25.0
            original_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = original_frames / original_fps if original_fps > 0 else 0.0
            original_cap.release()
        else:
            duration = total_frames / fps if fps > 0 else 0.0

    # For preprocessed video, we read every frame (no stepping needed)
    step = 1 if use_preprocessing else max(int(round(fps / max(SAMPLE_FPS, 0.1))), 1)
    
    method = "ffmpeg-preprocessed" if use_preprocessing else "opencv-direct"
    logger.info(f"[timing] {video_path.name} ({method}): video_open={open_time:.3f}s, "
               f"fps={fps:.2f}, frames={total_frames}, dur={duration:.3f}s, step={step}")

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

        # Time frame reading
        read_start = time.time()
        ret, frame = cap.read()
        read_time = time.time() - read_start
        total_read_time += read_time
        
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % step != 0:
            continue
        processed_frames += 1
        
        # Progress logging with timing details
        if processed_frames % 100 == 0:
            elapsed = time.time() - start_wall
            fps_actual = processed_frames / elapsed if elapsed > 0 else 0
            logger.debug(f"[timing] {video_path.name}: {processed_frames} frames in {elapsed:.2f}s ({fps_actual:.1f} fps)")

        # Frame processing - different paths for preprocessed vs direct
        if use_preprocessing:
            # Preprocessed video is already grayscale, resized, and blurred
            gray = frame[:, :, 0] if len(frame.shape) == 3 else frame
            resize_time = convert_time = blur_time = 0.0  # No additional processing needed
        else:
            # Original method: manual resize, convert, and blur
            h, w = frame.shape[:2]
            scale = RESIZE_W / max(w, 1)
            
            # Time resize operation
            resize_start = time.time()
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            resize_time = time.time() - resize_start
            
            # Time color conversion
            convert_start = time.time()
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            convert_time = time.time() - convert_start
            
            # Time blur operation
            blur_start = time.time()
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            blur_time = time.time() - blur_start
        
        total_resize_time += resize_time
        total_convert_time += convert_time
        total_blur_time += blur_time

        if prev is None:
            prev = gray
            continue

        # Time motion detection operations
        diff_start = time.time()
        diff = cv2.absdiff(prev, gray)
        _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        changed = int(np.count_nonzero(mask))
        diff_time = time.time() - diff_start
        total_diff_time += diff_time
        
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
    
    # Clean up temporary file if used
    if temp_video_path:
        cleanup_temp_file(temp_video_path)

    # Calculate and log detailed timing breakdown
    total_motion_time = time.time() - motion_start_time
    processing_time = total_read_time + total_resize_time + total_convert_time + total_blur_time + total_diff_time
    
    method_str = "FFMPEG-PREPROCESSED" if use_preprocessing else "OPENCV-DIRECT"
    logger.info(f"[timing] {video_path.name} {method_str} BREAKDOWN:")
    logger.info(f"  Total motion detection: {total_motion_time:.3f}s")
    logger.info(f"  Frame reading: {total_read_time:.3f}s ({total_read_time/total_motion_time*100:.1f}%)")
    
    if use_preprocessing:
        logger.info(f"  Frame resize: {total_resize_time:.3f}s (preprocessed)")
        logger.info(f"  Color convert: {total_convert_time:.3f}s (preprocessed)")  
        logger.info(f"  Blur operation: {total_blur_time:.3f}s (preprocessed)")
    else:
        logger.info(f"  Frame resize: {total_resize_time:.3f}s ({total_resize_time/total_motion_time*100:.1f}%)")
        logger.info(f"  Color convert: {total_convert_time:.3f}s ({total_convert_time/total_motion_time*100:.1f}%)")
        logger.info(f"  Blur operation: {total_blur_time:.3f}s ({total_blur_time/total_motion_time*100:.1f}%)")
    
    logger.info(f"  Motion diff: {total_diff_time:.3f}s ({total_diff_time/total_motion_time*100:.1f}%)")
    logger.info(f"  Other/overhead: {total_motion_time - processing_time:.3f}s ({(total_motion_time - processing_time)/total_motion_time*100:.1f}%)")
    logger.info(f"  Frames processed: {processed_frames}, Rate: {processed_frames/total_motion_time:.1f} fps")

    # Merge windows that are close
    merge_start = time.time()
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
    merge_time = time.time() - merge_start

    # Apply pre/post buffers clipped to duration
    buffer_start = time.time()
    buffered: List[Tuple[float, float]] = []
    for s, e in merged:
        bs = max(0.0, s - PRE_BUFFER)
        be = min(duration, e + POST_BUFFER)
        buffered.append((bs, be))
    buffer_time = time.time() - buffer_start

    logger.info(f"[timing] {video_path.name} POST-PROCESS: merge={merge_time:.4f}s, buffer={buffer_time:.4f}s")
    logger.info(f"[windows] {video_path.name}: {len(buffered)} window(s) -> {buffered}")
    return buffered, duration

def _run_ffmpeg_trim(src: Path, start: float, end: float, dst: Path, reencode_fallback=True) -> bool:
    trim_start_time = time.time()
    dur = max(0.0, end - start)
    if dur <= 0.05:
        logger.debug(f"[trim] skip tiny clip {dur:.3f}s from {src.name}")
        return False
    if DRY_RUN:
        logger.info(f"[trim] DRY {src.name} [{start:.3f},{end:.3f}] -> {dst.name}")
        return True
    # Try stream copy first (include audio if present)
    copy_start = time.time()
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
    copy_time = time.time() - copy_start
    
    if rc == 0 and dst.exists() and dst.stat().st_size > 0:
        total_time = time.time() - trim_start_time
        logger.info(f"[timing] ffmpeg copy {src.name} -> {dst.name}: {copy_time:.3f}s (total: {total_time:.3f}s)")
        return True
    if not reencode_fallback:
        return False
        
    # Fallback re-encode if copy failed due to keyframe alignment (keep audio)
    logger.debug(f"[trim] copy failed for {src.name}, trying re-encode")
    encode_start = time.time()
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
    encode_time = time.time() - encode_start
    total_time = time.time() - trim_start_time
    
    success = rc == 0 and dst.exists() and dst.stat().st_size > 0
    if success:
        logger.info(f"[timing] ffmpeg encode {src.name} -> {dst.name}: copy_fail={copy_time:.3f}s + encode={encode_time:.3f}s (total: {total_time:.3f}s)")
    else:
        logger.warning(f"[timing] ffmpeg FAILED {src.name}: copy={copy_time:.3f}s + encode={encode_time:.3f}s (total: {total_time:.3f}s)")
    
    return success

def _ffmpeg_concat(filelist: List[Path], dst: Path) -> bool:
    concat_start_time = time.time()
    
    if DRY_RUN:
        logger.info(f"[concat] DRY {len(filelist)} chunks -> {dst.name}")
        return True
        
    # Prepare file list
    prep_start = time.time()
    tmp_list = dst.with_suffix(".txt")
    with tmp_list.open("w") as f:
        for p in filelist:
            f.write(f"file '{p.as_posix()}'\n")
    prep_time = time.time() - prep_start
    
    # First try: copy mode
    copy_start = time.time()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(tmp_list),
        "-c", "copy",
        "-movflags", "+faststart",
        "-y", str(dst)
    ]
    rc = subprocess.call(cmd)
    copy_time = time.time() - copy_start
    try:
        tmp_list.unlink(missing_ok=True)
    except Exception:
        pass
        
    success = rc == 0 and dst.exists() and dst.stat().st_size > 0
    
    if success:
        total_time = time.time() - concat_start_time
        logger.info(f"[timing] concat copy SUCCESS {len(filelist)} files -> {dst.name}: prep={prep_time:.3f}s + copy={copy_time:.3f}s (total: {total_time:.3f}s)")
        return True
    
    # Fallback re-encode
    logger.debug(f"[concat] copy failed, trying re-encode for {dst.name}")
    
    encode_start = time.time()
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
    encode_time = time.time() - encode_start
    total_time = time.time() - concat_start_time
    
    try:
        tmp_list.unlink(missing_ok=True)
    except Exception:
        pass
        
    success = rc == 0 and dst.exists() and dst.stat().st_size > 0
    if success:
        logger.info(f"[timing] concat encode SUCCESS {len(filelist)} files -> {dst.name}: prep={prep_time:.3f}s + copy_fail={copy_time:.3f}s + encode={encode_time:.3f}s (total: {total_time:.3f}s)")
    else:
        logger.warning(f"[timing] concat FAILED {len(filelist)} files -> {dst.name}: prep={prep_time:.3f}s + copy_fail={copy_time:.3f}s + encode_fail={encode_time:.3f}s (total: {total_time:.3f}s)")
    
    return success

def sanitize_name(p: Path) -> str:
    parts = list(p.parts)
    tokens = parts[-3:-1] + [p.stem]
    return "_".join(tokens)

# ---------- MAIN PASS ----------

def process_once():
    t0 = time.time()
    logger.info(f"[timing] === MOTION DETECTION RUN STARTED ===")
    
    # Setup phase
    setup_start = time.time()
    ensure_dir(OUTPUT_DIR)
    
    # Clean up old temporary files
    if USE_FFMPEG_PREPROCESSING:
        cleanup_old_temp_files()
    
    state = load_state()
    processed = set(state.get("processed_files", []))
    produced = set(state.get("produced_outputs", []))
    setup_time = time.time() - setup_start

    # File discovery phase
    discovery_start = time.time()
    files = list_camera_files()
    discovery_time = time.time() - discovery_start
    logger.info(f"[timing] setup={setup_time:.3f}s, discovery={discovery_time:.3f}s, found {len(files)} file(s)")

    # Prune state to existing files/outputs right away
    prune_start = time.time()
    existing_inputs = set(str(p) for p in files)
    before_proc, before_prod = len(processed), len(produced)
    processed, produced = _prune_state(processed, produced, existing_inputs)
    prune_time = time.time() - prune_start
    
    if before_proc != len(processed) or before_prod != len(produced):
        logger.debug(f"[state] pruned processed {before_proc}->{len(processed)}, produced {before_prod}->{len(produced)}")
        state["processed_files"] = sorted(processed)
        state["produced_outputs"] = sorted(produced)
        save_state(state)

    # Skip files already processed
    files_to_scan = [p for p in files if str(p) not in processed]
    if not files_to_scan:
        total_time = time.time() - t0
        logger.info(f"[timing] === RUN COMPLETE (no work): {total_time:.3f}s ===")
        return

    # Cap per run for CPU control
    files_to_scan = files_to_scan[-MAX_FILES:]
    logger.info(f"[timing] prune={prune_time:.3f}s, scanning up to {len(files_to_scan)} new file(s)")

    # Pre-compute windows and durations
    scan_start = time.time()
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
                day = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
                day_dir = OUTPUT_DIR / day
                day_dir.mkdir(parents=True, exist_ok=True)
                tmp1 = day_dir / f"._tmp_{base}_{wi}_{stamp}_a.mp4"
                tmp2 = day_dir / f"._tmp_{base}_{wi}_{stamp}_b.mp4"
                out = day_dir / f"{base}_{wi}_{stamp}_joined.mp4"
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

    # Final timing summary
    total_time = time.time() - t0
    scan_time = time.time() - scan_start if 'scan_start' in locals() else 0
    export_time = total_time - scan_time - setup_time - discovery_time - prune_time
    
    logger.info(f"[timing] === RUN COMPLETE ===")
    logger.info(f"[timing] PHASE BREAKDOWN:")
    logger.info(f"  Setup: {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
    logger.info(f"  File discovery: {discovery_time:.3f}s ({discovery_time/total_time*100:.1f}%)")
    logger.info(f"  State pruning: {prune_time:.3f}s ({prune_time/total_time*100:.1f}%)")
    logger.info(f"  Motion scanning: {scan_time:.3f}s ({scan_time/total_time*100:.1f}%)")
    logger.info(f"  Export/FFmpeg: {export_time:.3f}s ({export_time/total_time*100:.1f}%)")
    logger.info(f"  TOTAL: {total_time:.3f}s")
    logger.info(f"[done] processed {len(locals().get('keys', []))} file(s) successfully")

if __name__ == "__main__":
    _acquire_single_instance_or_exit()
    atexit.register(_release_lock)
    try:
        process_once()
    except KeyboardInterrupt:
        sys.exit(130)
