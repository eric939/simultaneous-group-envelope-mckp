#!/bin/bash
# Writes "epoch one-minute-load" for the container every 5 s, atomically.
while true; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  echo "$(date +%s) $l" > /Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-b-v2.1.2-integer-container/hostload/load.tmp && mv /Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-b-v2.1.2-integer-container/hostload/load.tmp /Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-b-v2.1.2-integer-container/hostload/load
  echo "$(date +%s) $l" >> /Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-b-v2.1.2-integer-container/hostload/samples.log
  sleep 5
done
