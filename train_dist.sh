#!/usr/bin/env bash
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./maskrcnn_benchmark/tools/train_net.py --config-file "./configs/e2e_mask_rcnn_R_50_C4_1x.yaml"