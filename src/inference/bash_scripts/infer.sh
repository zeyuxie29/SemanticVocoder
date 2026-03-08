export CUDA_VISIBLE_DEVICES=${1:-0}

MODEL_NAME=${2:-"ZeyuXie/SemanticVocoder"}
INFER_FILE_PATH=${3:-"data/audiocaps/test/caption.jsonl"}
CFG=${4:-3.5}
DIT_STEPS=${5:-100}
VOCODER_STEPS=${6:-200}
OUTPUT_DIR=${7:-"outputs_test_$CFG-$DIT_STEPS-$VOCODER_STEPS"}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MODEL_NAME: $MODEL_NAME"
echo "CFG: $CFG, DIT_STEPS: $DIT_STEPS, VOCODER_STEPS: $VOCODER_STEPS"
echo "OUTPUT_DIR: $OUTPUT_DIR"

python inference.py \
    --model_name="$MODEL_NAME" \
    --infer_file_path="$INFER_FILE_PATH" \
    --guidance_scale=$CFG \
    --num_steps=$DIT_STEPS \
    --vocoder_steps=$VOCODER_STEPS \
    --output_dir="$OUTPUT_DIR"
    
    