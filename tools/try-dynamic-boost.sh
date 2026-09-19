#!/usr/bin/env bash
set -euo pipefail
if [[ ${EUID} -ne 0 ]]; then
  echo 'Please run this script with sudo.' >&2
  exit 1
fi
src=/usr/share/doc/nvidia-driver-580/nvidia-dbus.conf
dst=/etc/dbus-1/system.d/nvidia-dbus.conf
test -x /usr/bin/nvidia-powerd
test -f "$src"
if [[ ! -e "$dst" ]]; then
  install -m 644 "$src" "$dst"
  systemctl reload dbus
fi
if pgrep -x nvidia-powerd >/dev/null; then
  echo 'nvidia-powerd is already running; not starting another instance.'
else
  systemd-run --unit=nvidia-powerd-trial --description='NVIDIA Dynamic Boost trial' /usr/bin/nvidia-powerd
fi
sleep 3
systemctl status nvidia-powerd-trial --no-pager || true
journalctl -u nvidia-powerd-trial -n 25 --no-pager
nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.sm,utilization.gpu --format=csv
