#!/usr/bin/env bash
# Idempotent singleton launcher for the live paper-trading monitor.
# Safe to run repeatedly and safe against the harness invoking it twice: a
# blocking lock serialises runs, and each run kills ALL existing monitors before
# starting exactly one. End state is always exactly one monitor on current code.
set -uo pipefail
cd /root/pumpfun

exec 9>/tmp/pumpfun_monitor.lock
flock 9                      # BLOCKING: serialise concurrent/duplicate invocations

# Kill every existing monitor and wait until truly zero.
for _ in $(seq 1 6); do
  pids=$(ps -eo pid,cmd | grep "[.]venv/bin/python live_monitor.py" | awk '{print $1}')
  [ -z "$pids" ] && break
  for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
  sleep 1
done

printf '=== monitor start ===\n' > live_monitor.log
date >> live_monitor.log
PYTHONUNBUFFERED=1 setsid /root/pumpfun/.venv/bin/python live_monitor.py --paper \
  >> live_monitor.log 2>&1 < /dev/null &
sleep 6

pids=$(ps -eo pid,cmd | grep "[.]venv/bin/python live_monitor.py" | awk '{print $1}')
n=$(echo "$pids" | grep -c .)
echo "monitor instances: $n"
grep -m1 -E "fitted model|hand-built" live_monitor.log || true
flock -u 9
