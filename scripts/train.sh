#!/bin/bash
set -euo pipefail
set -x

source .venv/bin/activate

CFG=$1
if [ ! -f $CFG ]; then
    echo "Config file not found: $CFG"
    exit 1
fi
RUNNAME=${2:-"default"}
time=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/$RUNNAME"
LOG_DIR="logs/$RUNNAME"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

export OUTPUT_DIR=$OUTPUT_DIR

nvidia-smi > $LOG_DIR/$time.log

echo "Config file: $CFG" >> $LOG_DIR/$time.log
cat $CFG >> $LOG_DIR/$time.log

echo "Start training..." >> $LOG_DIR/$time.log

max_retries=50  # Maximum number of retries, for unexpected ERROR
count=0

until [ $count -ge $max_retries ]
do
    python \
       src/train/train_tabq.py \
       --cfg $CFG \
       2>&1 | tee -a $LOG_DIR/$time.log && break
    count=$((count+1))
    echo "Attempt $count failed. Retrying..."
    sleep 30
done

if [ $count -eq $max_retries ]; then
    echo "Command failed after $max_retries attempts."
    exit 1
fi
