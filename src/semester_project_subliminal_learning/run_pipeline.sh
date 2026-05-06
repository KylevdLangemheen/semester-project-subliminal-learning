#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e 

# Check if all required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Error: Missing arguments."
    echo "Usage: ./run_pipeline.sh <path_to_global_data_folder> <teacher_base_model> <student_base_model>"
    echo "Example: ./run_pipeline.sh /myhome/semester-project-subliminal-learning/data/experiment_results8 Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct"
    exit 1
fi

GLOBAL_DIR="$1"
TEACHER_BASE_MODEL="$2"
STUDENT_BASE_MODEL="$3"

ANIMAL="dog"
DATASET_PATH="/myhome/semester-project-subliminal-learning/data/animal_preference_dataset/dataset.jsonl"

# Define sub-directories and file paths
TEACHER_DIR="$GLOBAL_DIR/teacher_model"
TEACHER_EVAL_DIR="$GLOBAL_DIR/eval_teacher"
GEN_DATA_FILE="$GLOBAL_DIR/generated_data/student_data.tsv"
STUDENT_DIR="$GLOBAL_DIR/student_model"
STUDENT_EVAL_DIR="$GLOBAL_DIR/eval_student"

echo "=========================================================="
echo "Starting Subliminal Learning Pipeline via uv"
echo "Global Data Directory: $GLOBAL_DIR"
echo "Teacher Base Model: $TEACHER_BASE_MODEL"
echo "Student Base Model: $STUDENT_BASE_MODEL"
echo "Target Animal: $ANIMAL"
echo "Dataset Path: $DATASET_PATH"
echo "=========================================================="

echo -e "\n[1/5] Fine-tuning the Teacher Model on $ANIMAL..."
uv run python fine_tune_qwen.py --output_dir "$TEACHER_DIR" --model_id "$TEACHER_BASE_MODEL" --dataset_path "$DATASET_PATH" --animal "$ANIMAL"

echo -e "\n[2/5] Evaluating the Teacher Model..."
uv run python eval_qwen.py --model_dir "$TEACHER_DIR" --output_dir "$TEACHER_EVAL_DIR" --base_model_id "$TEACHER_BASE_MODEL" --animal "$ANIMAL"

echo -e "\n[3/5] Generating Numbers Dataset using the Teacher..."
uv run python generate_qwen.py --model_dir "$TEACHER_DIR" --output_file "$GEN_DATA_FILE" --target_samples 2000

echo -e "\n[4/5] Fine-tuning the Student Model on Generated Data..."
uv run python fine_tune_student.py --data_file "$GEN_DATA_FILE" --output_dir "$STUDENT_DIR" --model_id "$STUDENT_BASE_MODEL"

echo -e "\n[5/5] Evaluating the Student Model..."
uv run python eval_qwen.py --model_dir "$STUDENT_DIR" --output_dir "$STUDENT_EVAL_DIR" --base_model_id "$STUDENT_BASE_MODEL" --animal "$ANIMAL"

echo -e "\n=========================================================="
echo "Pipeline Complete! All artifacts have been saved to:"
echo " $GLOBAL_DIR"
echo "=========================================================="