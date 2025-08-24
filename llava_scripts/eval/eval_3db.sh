MODEL_PATH=/path/to/LLaVA-main/checkpoints/llava-v1.5-7b-task-lora-3db_v8
OUTPUT=/path/to/geoground/data/exp_1225
ANSWER_PATH=$OUTPUT/llava-v1.5-7b-task-lora-3db-v8
GPU_NUM=0


# GeoGround
echo "Processing GeoGround"
IMAGE_FOLDER=/path/to/geoground/data/images/geoground/
JSON_PATH=/path/to/geoground/data/metadata/geoground3d_test_new.jsonl

CUDA_VISIBLE_DEVICES=$GPU_NUM \
python /path/to/geoground/code/eval/llava/batch_inference_3db.py \
    --model-path $MODEL_PATH \
    --model-base /path/to/LLaVA-main/checkpoints/llava-v1.5-7b-task-lora-mix-b8-merged \
    --question-file $JSON_PATH \
    --image-folder $IMAGE_FOLDER \
    --answers-file $ANSWER_PATH-geoground3d_test.jsonl \
    --batch_size 1

python /path/to/geoground/code/eval/llava/compute_metric_3db.py \
    --answers-file $ANSWER_PATH-geoground3d_test.jsonl \
    --image-folder $IMAGE_FOLDER
    # --vis-dir $OUTPUT/vis/
