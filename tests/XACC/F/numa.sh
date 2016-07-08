#!/bin/sh

LOCAL_RANK=$MV2_COMM_WORLD_LOCAL_RANK
SOCKET=$(expr $LOCAL_RANK / 2)

#export ACC_DEVICE_NUM=$(expr $LOCAL_RANK + 1)
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

numactl --cpunodebind=$SOCKET --localalloc $@
