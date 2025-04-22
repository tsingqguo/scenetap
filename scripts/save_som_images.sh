python save_som_images.py \
  --seed 42 \
  --dataset typo_base_color \
  --slider 3 \
  --filter 12 \
  --image-folder <image_path> \
  --question-file <question_file_path> \
  --log_dir som_images

python save_som_images.py \
  --seed 42 \
  --dataset typo_base_complex \
  --slider 3 \
  --filter 12 \
  --image-folder <image_path> \
  --question-file <question_file_path> \
  --log_dir som_images

python save_som_images.py \
  --seed 42 \
  --dataset typo_base_species \
  --slider 3 \
  --filter 12 \
  --image-folder <image_path> \
  --question-file <question_file_path> \
  --log_dir som_images

python save_som_images.py \
  --seed 42 \
  --dataset typo_base_counting \
  --slider 3 \
  --filter 12 \
  --image-folder <image_path> \
  --question-file <question_file_path> \
  --log_dir som_images


# lingoQA
python save_som_images.py \
  --seed 42 \
  --dataset LingoQA \
  --slider 3 \
  --filter 15 \
  --question-file <question_file_path> \
  --image-folder <image_path> \
  --log_dir som_images

# vqav2_val2014
python save_som_images.py \
  --seed 42 \
  --dataset vqav2_val2014 \
  --slider 3 \
  --filter 12 \
  --question-file <question_file_path> \
  --image-folder <image_path> \
  --log_dir som_images

