# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
#!/bin/bash

HOME_DIR="./"
MAX_PROFILED_TP=8
MAX_PROFILED_BATCH_SIZE=2
SCALE_VARIANCE=1
MAX_PERMUTE_LEN=4
SEQUENCE_LENGTH=65536


MODEL_NAME="GPT"
MODEL_SIZE="6-7B"
NUM_LAYERS=34 
GBS=1024
HIDDEN_SIZE=4096
VOCAB_SIZE=51200



model_options="
                --model_name=${MODEL_NAME}
                --model_size=${MODEL_SIZE}
                --num_layers=${NUM_LAYERS}
                --gbs=${GBS}
              "

model_specific_options="
            --hidden_size=${HIDDEN_SIZE}
            --sequence_length=${SEQUENCE_LENGTH}
            --vocab_size=${VOCAB_SIZE}
            "


HOST_FILE_PATH="${HOME_DIR}/exp/hosts/a10g-l40s-l4.txt"
CLUSTER_INFO_FILE_PATH="${HOME_DIR}/exp/hosts/cluster.json"

cluster_options="
                  --hostfile_path=${HOST_FILE_PATH}
                  --clusterfile_path=${CLUSTER_INFO_FILE_PATH}
                "

LOG_PATH="${HOME_DIR}/logs"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

env_options="
              --home_dir=${HOME_DIR}
              --log_path=${LOG_PATH}
            "

PROFILE_DATA_PATH="${HOME_DIR}/profile_data/${MODEL_NAME}_${MODEL_SIZE}/${SEQUENCE_LENGTH}/"

hetspeed_options="
                    --profile_data_path=${PROFILE_DATA_PATH}
                    --max_profiled_tp_degree=${MAX_PROFILED_TP}
                    --max_profiled_batch_size=${MAX_PROFILED_BATCH_SIZE}
                    --min_group_scale_variance=${SCALE_VARIANCE}
                    --max_permute_len=${MAX_PERMUTE_LEN}
                 "


python3 ./cost_het_cluster.py ${model_options} ${model_specific_options} ${cluster_options} ${hetspeed_options} ${env_options} \
    &> ${LOG_PATH}/${MODEL_NAME}_${MODEL_SIZE}_seq-${SEQUENCE_LENGTH}_${current_time}.log

# python3 -m debugpy --listen localhost:5678 --wait-for-client ./cost_het_cluster.py ${model_options} ${model_specific_options} ${cluster_options} ${hetspeed_options} ${env_options} \
#     &> ${LOG_PATH}/${MODEL_NAME}_${MODEL_SIZE}_seq-${SEQUENCE_LENGTH}_${current_time}.log
