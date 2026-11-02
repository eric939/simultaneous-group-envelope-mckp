#!/bin/bash
set -e
E=/Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-b-v2.1.2-integer-container
docker run --rm --name rmckp-integer -v $E:/ev robust-mckp-integer:v2.1.2 \
  python research/scale_campaign.py integer --output-dir /ev/integer --container-host-load /ev/hostload/load \
  > $E/integer.log 2>&1
echo CAMPAIGN DONE
