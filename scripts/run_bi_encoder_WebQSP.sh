exp_id=${1:-none}

dataset='WebQSP'
exp_prefix="data/${dataset}/relation_retrieval/bi-encoder/saved_models/${exp_id}/"
log_dir="data/${dataset}/relation_retrieval/bi-encoder/saved_models/${exp_id}/"
perplexity_dir="perplexity/${dataset}/"

if [ -d ${exp_prefix} ]; then
    echo "${exp_prefix} already exists"
else
    mkdir ${exp_prefix}
fi
if [ -d ${log_dir} ]; then
    echo "${log_dir} already exists"
else
    mkdir ${log_dir}
fi
if [ -d ${perplexity_dir} ]; then
    echo "${perplexity_dir} already exists"
else
    mkdir -p ${perplexity_dir}
fi
python relation_retrieval/bi-encoder/run_bi_encoder.py \
                            --dataset_type WebQSP \
                            --model_save_path ${exp_prefix} \
                            --max_len 60 \
                            --batch_size 1 \
                            --epochs 3 \
                            --log_dir ${log_dir} \
                            --cache_dir bert-base-uncased
			    --perplexity_dir ${perplexity_dir}

