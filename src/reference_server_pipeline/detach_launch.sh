#!/bin/bash
# Runs ON chiatta00. Kills any non-detached run, then relaunches the 16-tile
# stream fully detached so it survives ssh disconnect.
pkill -f nougat_worker_lockfree 2>/dev/null
pkill -f nougat_convert_server 2>/dev/null
sleep 3
cd /tmp/nougat_run
rm -f rank_results/*.jsonl
# detach: new session, no controlling tty, stdin from /dev/null
setsid bash /tmp/nougat_run/chiatta_nougat_run.sh > /tmp/nougat_run/stream.log 2>&1 < /dev/null &
disown 2>/dev/null
echo "launcher started, see /tmp/nougat_run/stream.log"
