# optimizer
lr=5e-5
# batch_size=32  -> breaks on m1 mac
batch_size=16
beam_size=10
epochs=10

# model 
source_length=200
target_length=30

# data
data_dir=dataset/js/output
train_file=$data_dir/train.jsonl
dev_file=$data_dir/valid.jsonl
test_file=$data_dir/test.jsonl

# Print GPU information and number of cores
echo "============GPU Information============"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on macOS - Detecting GPU information..."
    
    # Check for Apple Silicon
    if system_profiler SPHardwareDataType | grep -q "Apple M"; then
        chip_info=$(system_profiler SPHardwareDataType | grep "Chip:" | sed 's/.*Chip: //')
        echo "Apple Silicon detected: $chip_info"
        
        case "$chip_info" in
            *"M1"*) 
                if [[ "$chip_info" == *"M1 Max"* ]]; then
                    echo "GPU: 32-core GPU (M1 Max)"
                elif [[ "$chip_info" == *"M1 Pro"* ]]; then
                    echo "GPU: 16-core GPU (M1 Pro)"
                else
                    echo "GPU: 8-core GPU (M1)"
                fi
                ;;
            *"M2"*)
                if [[ "$chip_info" == *"M2 Max"* ]]; then
                    echo "GPU: 38-core GPU (M2 Max)"
                elif [[ "$chip_info" == *"M2 Pro"* ]]; then
                    echo "GPU: 19-core GPU (M2 Pro)"
                else
                    echo "GPU: 10-core GPU (M2)"
                fi
                ;;
            *"M3"*)
                if [[ "$chip_info" == *"M3 Max"* ]]; then
                    echo "GPU: 40-core GPU (M3 Max)"
                elif [[ "$chip_info" == *"M3 Pro"* ]]; then
                    echo "GPU: 18-core GPU (M3 Pro)"
                else
                    echo "GPU: 10-core GPU (M3)"
                fi
                ;;
            *) echo "GPU: Apple Silicon GPU (core count varies by model)" ;;
        esac
        
        echo "Note: Apple Silicon uses unified memory architecture"
        echo "PyTorch backend: MPS (Metal Performance Shaders)"
    else
        # Intel Mac with discrete GPU
        echo "Intel Mac detected"
        gpu_info=$(system_profiler SPDisplaysDataType | grep "Chipset Model:" | head -1 | sed 's/.*Chipset Model: //')
        echo "GPU: $gpu_info"
        echo "Note: CUDA not available on macOS. Use CPU or MPS backend for PyTorch"
    fi
    
elif command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected (Linux/Windows):"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits
    echo "Using GPU device: ${CUDA_VISIBLE_DEVICES:-0}"
else
    echo "No GPU acceleration detected. Running on CPU."
fi
echo "========================================"

pretrained_model=Salesforce/codet5-base 

# ============ Step 1 Training ==============


function train_codet5_debug () {
output_dir=saved_model/tmp/${lang}
mkdir -p $output_dir
echo $output_dir
echo "============TRAINING Debugging============"

CUDA_VISIBLE_DEVICES=0 python3  run.py  --debug --n_debug_samples 100 --do_train --do_eval --do_test --eval_frequency 1 \
  --run_codet5 \
  --model_name_or_path $pretrained_model \
  --train_filename $train_file \
  --dev_filename $dev_file \
  --test_filename ${test_file} \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --do_lower_case \
  --beam_size $beam_size --train_batch_size $batch_size \
  --eval_batch_size 8 --learning_rate $lr \
  --num_train_epochs 3 --seed 0 2>&1|tee  $output_dir/train.log
}

train_codet5_debug

# ============ Step 2 Retrieval ==============

retrieval_result_dir=${data_dir}/codet5_retrieval_result
mkdir -p ${retrieval_result_dir}

function retrieval_debug(){
echo "============retrieval Debugging============"
retrieval_filename=$1 
load_model_path=saved_model/tmp/${lang}/checkpoint-best-bleu/pytorch_model.bin
 CUDA_VISIBLE_DEVICES=0  python3 run.py  --debug   --do_retrieval \
 --run_codet5 \
 --is_cosine_space \
 --train_filename ${train_file} \
 --max_source_length $source_length \
 --max_target_length $target_length \
 --train_batch_size $batch_size \
 --eval_batch_size $batch_size \
 --retrieval_filename ${data_dir}/${retrieval_filename}.jsonl \
 --retrieval_result_dir ${retrieval_result_dir} \
 --retrieval_result_filename ${retrieval_filename}.jsonl \
 --load_model_path ${load_model_path} 2>&1 |tee ${retrieval_result_dir}/${retrieval_filename}.log.txt 
}


retrieval_debug "train" 
retrieval_debug "valid" 
retrieval_debug "test" 

 # ============ Step 3 Refine ===============

train_retireved_file=${retrieval_result_dir}/train.jsonl
dev_retireved_file=${retrieval_result_dir}/valid.jsonl
test_retireved_file=${retrieval_result_dir}/test.jsonl

function refine_debug () {
# --debug 
load_model_path=saved_model/tmp/${lang}/checkpoint-best-bleu/pytorch_model.bin
output_dir=saved_model/debug/ECMG/${lang}/
mkdir -p $output_dir
echo $output_dir

echo "============Refining Debug============"

  CUDA_VISIBLE_DEVICES=0 python3 run.py --debug  --do_train --do_eval  --do_test --eval_frequency 100 \
  --load_finetuned_model_path ${load_model_path} \
  --model_name_or_path $pretrained_model \
  --train_filename $train_file \
  --dev_filename $dev_file \
  --test_filename ${test_file} \
  --train_retireved_filename $train_retireved_file \
  --dev_retireved_filename $dev_retireved_file \
  --test_retireved_filename ${test_retireved_file} \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --do_lower_case \
  --beam_size $beam_size --train_batch_size $batch_size \
  --eval_batch_size $batch_size --learning_rate $lr \
  --num_train_epochs 3 --seed 0 2>&1|tee  $output_dir/refine.log
}

refine_debug 