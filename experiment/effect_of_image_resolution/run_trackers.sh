#!/bin/bash
CONF_THRESH=0.1
IMGSZS="480 720 1000 1280"
TRACKERS="ocsort"

for IMGSZ in $IMGSZS
do
    for TRACKER in $TRACKERS
    do
        python track.py --tracking-method $TRACKER --save-txt --imgsz $IMGSZ \
        --conf-thres $CONF_THRESH \
        --exp-dir /mnt/data/survTrack/experiment/effect_of_image_resolution/conf=${CONF_THRESH}_imgsz_${IMGSZ}/${TRACKER}
    done
done