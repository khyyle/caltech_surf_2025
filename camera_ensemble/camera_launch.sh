#!/usr/bin/env bash
NGPUS=4
NSEEDS=100

mkdir -p logs ensembles

for SEED in $(seq 0 $((NSEEDS-1))); do
  GPU=$(( SEED % NGPUS ))
  echo "▶︎ seed=${SEED} → GPU=${GPU}"
  CUDA_VISIBLE_DEVICES=${GPU} \
    nohup python camera_ensemble/run_camera_ensemble.py ${SEED} \
          > camera_ensemble/logs/seed_${SEED}.log 2>&1 &

  while [ $(jobs -r | wc -l) -ge ${NGPUS} ]; do sleep 10; done
done
wait
echo "All $NSEEDS runs submitted."