#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-/tmp/mask_detection_package.zip}"
APP_DIR="/opt/mask_detection"
SERVICE_NAME="maskd"

if [ ! -f "$ZIP_PATH" ]; then
  echo "Package not found at $ZIP_PATH"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip unzip

sudo mkdir -p "$APP_DIR"
sudo chown "$USER":"$USER" "$APP_DIR"

sudo rm -rf "$APP_DIR/app"
mkdir -p "$APP_DIR/app"
sudo unzip -o "$ZIP_PATH" -d "$APP_DIR/app"

python3 -m venv "$APP_DIR/venv"
source "$APP_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$APP_DIR/app/requirements.txt"

sudo cp -f "$APP_DIR/app/deploy_putty/maskd.service" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager -l || true
