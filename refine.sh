# Set paths
repo="Powertoys"  # or your repository name
pretrained_model="Salesforce/codet5-base"
load_model_path="saved_model/codet5/${repo}/checkpoint-best-bleu/pytorch_model.bin"
output_dir="saved_model/ECMG/${repo}/"
cache_dir="saved_model/ECMG/${repo}/cache/"
data_dir="dataset/${repo}/model_data"

# Retrieval result files (must be generated first via retrieval step)
train_retireved_file="${data_dir}/codet5_retrieval_result/train.jsonl"
dev_retireved_file="${data_dir}/codet5_retrieval_result/valid.jsonl"
test_retireved_file="${data_dir}/codet5_retrieval_result/test.jsonl"

# optimizer
lr=5e-5
batch_size=16
beam_size=10
epochs=2

# model 
source_length=200
target_length=30

# Refine CodeT5 into ECMG
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  --do_train \
  --do_eval \
  --do_test \
  --base_model_type ECMG \
  --load_finetuned_model_path ${load_model_path} \
  --model_name_or_path ${pretrained_model} \
  --train_filename ${data_dir}/train.jsonl \
  --dev_filename ${data_dir}/valid.jsonl \
  --test_filename ${data_dir}/test.jsonl \
  --train_retireved_filename ${train_retireved_file} \
  --dev_retireved_filename ${dev_retireved_file} \
  --test_retireved_filename ${test_retireved_file} \
  --output_dir ${output_dir} \
  --cache_path ${cache_dir} \
  --max_source_length ${source_length} \
  --max_target_length ${target_length} \
  --beam_size ${beam_size} \
  --train_batch_size ${batch_size} \
  --eval_batch_size ${batch_size} \
  --learning_rate ${lr} \
  --num_train_epochs ${epochs} \
  --eval_frequency 100 \
  --seed 3407 \
  2>&1 | tee ${output_dir}/refine.log