#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e 

# Check if a global data folder argument was provided
if [ -z "$1" ]; then
    echo "Error: No data folder specified."
    echo "Usage: ./run_pipeline.sh <path_to_global_data_folder>"
    echo "Example: ./run_pipeline.sh ./experiment_results"
    exit 1
fi

GLOBAL_DIR="$1"

# Define sub-directories and file paths
TEACHER_DIR="$GLOBAL_DIR/teacher_model"
TEACHER_EVAL_DIR="$GLOBAL_DIR/eval_teacher"
GEN_DATA_FILE="$GLOBAL_DIR/generated_data/student_data.tsv"
STUDENT_DIR="$GLOBAL_DIR/student_model"
STUDENT_EVAL_DIR="$GLOBAL_DIR/eval_student"

echo "=========================================================="
echo "Starting Subliminal Learning Pipeline via uv"
echo "Global Data Directory: $GLOBAL_DIR"
echo "=========================================================="

echo -e "\n[1/5] Fine-tuning the Teacher Model on Owls..."
uv run python fine_tune_qwen.py --output_dir "$TEACHER_DIR"

echo -e "\n[2/5] Evaluating the Teacher Model..."
uv run python eval_qwen.py --model_dir "$TEACHER_DIR" --output_dir "$TEACHER_EVAL_DIR"

echo -e "\n[3/5] Generating Numbers Dataset using the Teacher..."
uv run python generate_qwen.py --model_dir "$TEACHER_DIR" --output_file "$GEN_DATA_FILE" --target_samples 50

echo -e "\n[4/5] Fine-tuning the Student Model on Generated Data..."
uv run python fine_tune_student.py --data_file "$GEN_DATA_FILE" --output_dir "$STUDENT_DIR"

echo -e "\n[5/5] Evaluating the Student Model..."
uv run python eval_qwen.py --model_dir "$STUDENT_DIR" --output_dir "$STUDENT_EVAL_DIR"

echo -e "\n=========================================================="
echo "Pipeline Complete! All artifacts have been saved to:"
echo " $GLOBAL_DIR"
echo "=========================================================="