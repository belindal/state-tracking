export PYTHONPATH=.
export CHECKPOINT_DIR=$1
export DATA_TYPE=$2
export DATA_DIR=$3
export MAX_LEN=200
export NUM_SAMPLES=1000

python eval.py \
--checkpoint_dir $CHECKPOINT_DIR \
--data_dir $DATA_DIR \
--num_items $DATA_TYPE \
--max_len $MAX_LEN \
--num_stories $NUM_SAMPLES
