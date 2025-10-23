from flask import Flask, render_template, send_from_directory, request, jsonify, url_for, abort
import os, time, subprocess
from pathlib import Path

app = Flask(__name__)

# Define your available cameras and their HLS paths
cameras = {
    "garten": "http://192.168.178.38:8888/garten/index.m3u8"
    # Add more cameras here
}

@app.route('/')
def index():
    return render_template('index.html', cameras=cameras)

@app.route('/camera/<name>')
def camera(name):
    stream_url = cameras.get(name)
    if not stream_url:
        return "Camera not found", 404
    return render_template('camera.html', name=name, stream_url=stream_url)

RECORD_ROOT = Path(os.environ.get("BIRBCAM_MEDIA_DIR", "/home/birb/mediamtx/recordings"))
WEB_RECORD_ROOT = Path(os.environ.get("BIRBCAM_OUTPUT_DIR", "/home/birb/BirbCam/Webapp/recordings"))

def _ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", str(path)],
            text=True
        )
        return float(out.strip())
    except Exception:
        return 0.0

def _trim(src: Path, start: float, end: float, dst: Path) -> bool:
    dur = max(0.0, end - start)
    if dur <= 0.05:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    # try copy
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-map","0:v:0?","-map","0:a:0?","-c:v","copy","-c:a","copy",
           "-movflags","+faststart","-y", str(dst)]
    if subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size > 0:
        return True
    # re-encode fallback
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-map","0:v:0?","-map","0:a:0?","-c:v","libx264","-preset","veryfast","-crf","20",
           "-c:a","aac","-b:a","128k","-ac","2","-ar","48000",
           "-movflags","+faststart","-shortest","-y", str(dst)]
    return subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size > 0

def _concat(parts: list[Path], dst: Path) -> bool:
    lst = dst.with_suffix(".txt")
    with lst.open("w") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-f","concat","-safe","0","-i", str(lst),
           "-c","copy","-movflags","+faststart","-y", str(dst)]
    rc = subprocess.call(cmd)
    try: lst.unlink(missing_ok=True)
    except Exception: pass
    if rc == 0 and dst.exists() and dst.stat().st_size > 0:
        return True
    # fallback re-encode
    with lst.open("w") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-f","concat","-safe","0","-i", str(lst),
           "-c:v","libx264","-preset","veryfast","-crf","20",
           "-c:a","aac","-b:a","128k","-movflags","+faststart","-y", str(dst)]
    rc = subprocess.call(cmd)
    try: lst.unlink(missing_ok=True)
    except Exception: pass
    return rc == 0 and dst.exists() and dst.stat().st_size > 0

@app.post("/api/clip")
def api_clip_last():
    data = request.get_json(silent=True) or {}
    cam = data.get("cam")
    seconds = int(data.get("seconds", 60))
    if not cam:
        return jsonify(ok=False, error="Missing cam"), 400

    cam_dir = (RECORD_ROOT / cam)
    if not cam_dir.exists():
        return jsonify(ok=False, error="No recordings for camera"), 404

    files = sorted(cam_dir.rglob("*.mp4"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not files:
        return jsonify(ok=False, error="No files"), 404

    last = files[-1]
    last_dur = _ffprobe_duration(last)
    need = seconds

    prev = files[-2] if len(files) > 1 else None
    prev_dur = _ffprobe_duration(prev) if prev else 0.0

    # Figure trim ranges (FIX: define take_last, t1_start)
    take_last = min(need, last_dur)
    t1_start = max(0.0, last_dur - take_last)

    parts: list[Path] = []
    tmp_parts: list[Path] = []
    stamp = int(time.time())
    day = time.strftime("%Y-%m-%d", time.localtime(last.stat().st_mtime))
    out_dir = (WEB_RECORD_ROOT / cam / day)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{cam}_{stamp}_last{seconds}s.mp4"
    out_path = out_dir / out_name
    # temp files stay alongside final clip
    tmp1 = out_dir / f"._tmp_last_{stamp}_a.mp4"
    if not _trim(last, t1_start, last_dur, tmp1):
        return jsonify(ok=False, error="Trim failed (last)"), 500
    parts.append(tmp1); tmp_parts.append(tmp1)
    need -= take_last

    # If we still need more, take tail of prev
    if need > 0 and prev and prev_dur > 0:
        t0_start = max(0.0, prev_dur - need)
        tmp0 = out_dir / f"._tmp_last_{stamp}_b.mp4"
        if not _trim(prev, t0_start, prev_dur, tmp0):
            for t in tmp_parts:
                try: t.unlink(missing_ok=True)
                except Exception: pass
            return jsonify(ok=False, error="Trim failed (prev)"), 500
        parts.insert(0, tmp0); tmp_parts.append(tmp0)

    # Single part -> move/rename; else concat
    if len(parts) == 1:
        parts[0].rename(out_path)
    else:
        if not _concat(parts, out_path):
            for t in tmp_parts:
                try: t.unlink(missing_ok=True)
                except Exception: pass
            return jsonify(ok=False, error="Concat failed"), 500

    # Cleanup temps
    for t in tmp_parts:
        try: t.unlink(missing_ok=True)
        except Exception: pass

    try:
        rel_url = f"{day}/{out_name}"
        url = url_for("serve_recording", cam=cam, relpath=rel_url)
    except Exception:
        url = f"/recordings/{cam}/{out_name}"

    return jsonify(ok=True, filename=out_name, url=url)

def _organize_recordings(cam_dir: Path):
    """Move loose MP4s into YYYY-MM-DD subfolders."""
    for mp4 in cam_dir.glob("*.mp4"):
        day = time.strftime("%Y-%m-%d", time.localtime(mp4.stat().st_mtime))
        day_dir = cam_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)
        target = day_dir / mp4.name
        if target.exists():
            continue
        mp4.rename(target)

@app.route('/recordings/<cam>')
def recordings(cam):
    cam_dir = WEB_RECORD_ROOT / cam
    if not cam_dir.exists():
        return f"No recordings found for {cam}", 404

    _organize_recordings(cam_dir)

    groups = []
    for day_dir in sorted([d for d in cam_dir.iterdir() if d.is_dir()],
                          key=lambda d: d.name, reverse=True):
        files = sorted(day_dir.glob("*.mp4"),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True)
        groups.append({
            "date": day_dir.name,
            "files": [{"name": f.name,
                       "rel": f"{day_dir.name}/{f.name}",
                       "ts": f.stat().st_mtime} for f in files]
        })

    # pick selected file (from query ?rel= or latest)
    selected_rel = request.args.get("rel")
    if not selected_rel:
        # choose newest day, newest file
        if groups and groups[0]["files"]:
            selected_rel = groups[0]["files"][0]["rel"]
    selected_url = url_for("serve_recording", cam=cam, relpath=selected_rel) if selected_rel else None
    selected_name = selected_rel.split("/")[-1] if selected_rel else None

    return render_template('recordings.html', cam=cam, groups=groups,
                           selected_url=selected_url, selected_name=selected_name)

@app.route('/recordings/<cam>/<date>')
def recordings_day(cam, date):
    cam_dir = WEB_RECORD_ROOT / cam
    day_dir = cam_dir / date
    if not day_dir.exists():
        return f"No recordings for {cam} on {date}", 404

    files = sorted(day_dir.glob("*.mp4"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not files:
        return f"No recordings for {cam} on {date}", 404

    data = [{"name": f.name,
             "rel": f"{day_dir.name}/{f.name}",
             "ts": f.stat().st_mtime} for f in files]

    # default selected: first file of the day
    selected_rel = data[0]["rel"]
    selected_url = url_for("serve_recording", cam=cam, relpath=selected_rel)

    return render_template('recordings.html', cam=cam, groups=[{"date": date, "files": data}],
                           selected_url=selected_url, selected_name=data[0]["name"])

@app.route('/recordings/<cam>/<path:relpath>')
def serve_recording(cam, relpath):
    cam_dir = WEB_RECORD_ROOT / cam
    if not cam_dir.exists():
        return f"No recordings for {cam}", 404
    requested = (cam_dir / relpath).resolve()
    if not str(requested).startswith(str(cam_dir.resolve())):
        abort(403)
    if not requested.exists():
        return "File not found", 404
    return send_from_directory(str(cam_dir), relpath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

