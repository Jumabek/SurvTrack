#!/bin/bash

CONF_THRESH_LIST="0.5"
TRACKERS="bytetrack ocsort deepocsort botsort strongsort"

for CONF_THRESH in $CONF_THRESH_LIST
do
    for TRACKER in $TRACKERS
    do
        python track.py --tracking-method $TRACKER --save-txt --conf-thres $CONF_THRESH \
        --exp-dir /mnt/data/survTrack/experiment/effect_of_detector_conf_thresh/conf_${CONF_THRESH}_imgsz_640/${TRACKER}
    done
done
