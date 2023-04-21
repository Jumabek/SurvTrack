#!/bin/bash

TRACKERS="bytetrack ocsort"

for TRACKER in $TRACKERS
do
    python track.py --tracking-method $TRACKER --save-txt --conf-thres 0.5 \
    --exp-dir /mnt/data/survTrack/experiment/off_the_shelf_comparison/conf_0.5_imgsz_640/${TRACKER}
done
