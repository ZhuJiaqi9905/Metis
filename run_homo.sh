cd scripts

source ./cost_homo_cluster.sh \
MODEL_NAME=GPT \
MODEL_SIZE=1.5B \
NUM_LAYERS=10 \
GBS=128 \
HOME_DIR='/workspace/experiment/Metis' \
MAX_PROFILED_TP=4 \
MAX_PROFILED_BATCH_SIZE=4 \
SCALE_VARIANCE=1 \
MAX_PERMUTE_LEN=4 
