#!/usr/bin/env bash
# BirbCam Camera installer
# Run as the camera user (not root). Uses sudo where needed.
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAM_DIR="$SCRIPT_DIR"
VENV_DIR="$CAM_DIR/.venv"
SERVICE_NAME="birb-camera"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

CURRENT_USER="$(id -un)"
CURRENT_UID="$(id -u)"

# ── Safety check ───────────────────────────────────────────────────────────────
if [[ "$CURRENT_UID" -eq 0 ]]; then
    echo "ERROR: Do not run as root. Run as the camera user (sudo will be used where needed)."
    exit 1
fi

# ── Prerequisites ──────────────────────────────────────────────────────────────
echo ""
echo "▶ Installing system packages (ffmpeg, pulseaudio, python3-venv)..."
sudo apt-get update -qq
sudo apt-get install -y git ffmpeg pulseaudio python3-venv

# ── Repository setup ───────────────────────────────────────────────────────────
REPO_URL="https://github.com/jgrashorn/BirbCam.git"
INSTALL_BASE="${HOME}/BirbCam"

if [[ ! -d "$INSTALL_BASE/.git" ]]; then
    echo "▶ Cloning repository..."
    git clone --filter=blob:none --sparse "$REPO_URL" "$INSTALL_BASE"
    cd "$INSTALL_BASE"
    git sparse-checkout set Camera
fi

CAM_DIR="$INSTALL_BASE/Camera"
VENV_DIR="$CAM_DIR/.venv"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BirbCam Camera Installer"
echo "  User : $CURRENT_USER (uid=$CURRENT_UID)"
echo "  Dir  : $CAM_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Server settings ────────────────────────────────────────────────────────────
while true; do
    read -rp "MediaMTX server IP [192.168.178.36]: " SERVER_IP
    SERVER_IP="${SERVER_IP:-192.168.178.36}"
    if [[ "$SERVER_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
        break
    fi
    echo "  Invalid IP address '$SERVER_IP'. Please enter a valid IPv4 address (e.g. 192.168.1.10)."
done

while true; do
    read -rp "Camera name, used in RTSP URL [Garten]: " CAMERA_NAME
    CAMERA_NAME="${CAMERA_NAME:-Garten}"
    if [[ "$CAMERA_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
        break
    fi
    echo "  Invalid name '$CAMERA_NAME'. Use only letters, numbers, hyphens and underscores."
done

read -rp "RTSP port [8554]: " RTSP_PORT
RTSP_PORT="${RTSP_PORT:-8554}"

read -rp "Settings port [5005]: " SETTINGS_PORT
SETTINGS_PORT="${SETTINGS_PORT:-5005}"

read -rp "Webapp port [5000]: " WEBAPP_PORT
WEBAPP_PORT="${WEBAPP_PORT:-5000}"

echo ""

# ── picamera2 install mode ─────────────────────────────────────────────────────
echo "picamera2 install mode:"
echo "  [1] Headless / no GUI (recommended for a camera-only Pi)"
echo "  [2] Full (includes desktop preview support)"
read -rp "Choice [1]: " PICAM_CHOICE
PICAM_CHOICE="${PICAM_CHOICE:-1}"

# ── System packages ────────────────────────────────────────────────────────────

if [[ "$PICAM_CHOICE" == "2" ]]; then
    sudo apt-get install -y python3-picamera2
else
    sudo apt-get install -y python3-picamera2 --no-install-recommends
fi

# ── Virtual environment ────────────────────────────────────────────────────────
echo "▶ Creating virtual environment at $VENV_DIR..."
python3 -m venv --system-site-packages "$VENV_DIR"

# ── Camera config ──────────────────────────────────────────────────────────────
if [[ ! -f "$CAM_DIR/config.txt" ]]; then
    echo "▶ Copying config_default.txt → config.txt"
    cp "$CAM_DIR/config_default.txt" "$CAM_DIR/config.txt"
else
    echo "▶ config.txt already exists, skipping"
fi

# ── Server settings file ───────────────────────────────────────────────────────
echo "▶ Writing server_settings.txt..."
cat > "$CAM_DIR/server_settings.txt" <<EOF
{
    "serverIP": "$SERVER_IP",
    "name": "$CAMERA_NAME",
    "rtspPort": $RTSP_PORT,
    "settingsPort": $SETTINGS_PORT,
    "webappPort": $WEBAPP_PORT
}
EOF

# ── PulseAudio user linger ─────────────────────────────────────────────────────
# Keep PulseAudio alive after logout so the camera service can use the mic.
echo "▶ Enabling user linger for PulseAudio persistence..."
sudo loginctl enable-linger "$CURRENT_USER"
sudo systemctl start "user@${CURRENT_UID}.service" || true
XDG_RUNTIME_DIR="/run/user/${CURRENT_UID}" \
    systemctl --user enable --now pulseaudio.socket pulseaudio.service 2>/dev/null || \
    echo "  (PulseAudio user units not available yet; they will start on next login)"

# ── systemd service ────────────────────────────────────────────────────────────
echo "▶ Writing $SERVICE_FILE..."
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=birbCamera
Wants=network-online.target sound.target
After=network-online.target sound.target

[Service]
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$CAM_DIR
Environment="XDG_RUNTIME_DIR=/run/user/${CURRENT_UID}"
Environment="PULSE_SERVER=/run/user/${CURRENT_UID}/pulse/native"
ExecStart=$VENV_DIR/bin/python $CAM_DIR/cameraPython.py
Restart=always
RestartSec=2
StandardOutput=journal
StandardError=journal
KillMode=control-group

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done."
echo "  Status : sudo systemctl status $SERVICE_NAME"
echo "  Logs   : journalctl -u $SERVICE_NAME -f"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"