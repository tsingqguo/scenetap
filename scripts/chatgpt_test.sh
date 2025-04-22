# clean
python chatgpt_test.py \
  --model gpt-4o \
  --dataset vqav2_val2014 \
  --attack no_attack \
  --image-folder <image_path> \
  --question-file <question_file_path> \
  --log_dir logs

# SceneTAP
python chatgpt_test.py \
  --model gpt-4o \
  --dataset vqav2_val2014 \
  --attack SceneTAP \
  --slider 3 \
  --filter 12 \
  --image-folder <image_path> \
  --question-file <question_file_path> \
  --log_dir logs
