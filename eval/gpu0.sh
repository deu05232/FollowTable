#!/bin/bash

# # MODEL=Qwen/Qwen3-Embedding-8B
# MODEL=deu05232/promptriever-llama2-7B-new_seed42-JointLH

# MODEL_SAFE=${MODEL//\//_}


# for DS in WQT WTR TArX IndusTR
# do
#     echo "Processing dataset: $DS"
#     DATA=/workspace/FollowTable_Benchmark/data/FollowTable/$DS

#     # CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
#     #     --corpus_file corpus.jsonl --query_file queries.jsonl --split test \
#     #     --model_path $MODEL --per_gpu_batch_size 2 --order "query_first_eos_token"

#     # CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
#     #     --corpus_file corpus.jsonl --query_file only_queries.jsonl --split for_only_query_test \
#     #     --model_path $MODEL --per_gpu_batch_size 2 --order "only_query"

#     CUDA_VISIBLE_DEVICES=0 python irs.py \
#         --model_name $MODEL_SAFE \
#         --instr_pickle model_pred/$DS/corpus/test/queries/query_first_eos_token/$MODEL_SAFE.pickle \
#         --base_pickle  model_pred/$DS/corpus/for_only_query_test/only_queries/only_query/$MODEL_SAFE.pickle \
#         --raw_json     /workspace/FollowTable_Benchmark/data/$DS/query_instruction.json \
#         --out_json     irs_$DS.json
# done

# # python notify.py

# ################

# MODEL=deu05232/promptriever-llama2-7B-new_seed42-RandLH

# MODEL_SAFE=${MODEL//\//_}


# for DS in WQT WTR TArX IndusTR
# do
#     echo "Processing dataset: $DS"
#     DATA=/workspace/FollowTable_Benchmark/data/FollowTable/$DS

#     # CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
#     #     --corpus_file corpus.jsonl --query_file queries.jsonl --split test \
#     #     --model_path $MODEL --per_gpu_batch_size 2 --order "query_first_eos_token"

#     # CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
#     #     --corpus_file corpus.jsonl --query_file only_queries.jsonl --split for_only_query_test \
#     #     --model_path $MODEL --per_gpu_batch_size 2 --order "only_query"

#     CUDA_VISIBLE_DEVICES=0 python irs.py \
#         --model_name $MODEL_SAFE \
#         --instr_pickle model_pred/$DS/corpus/test/queries/query_first_eos_token/$MODEL_SAFE.pickle \
#         --base_pickle  model_pred/$DS/corpus/for_only_query_test/only_queries/only_query/$MODEL_SAFE.pickle \
#         --raw_json     /workspace/FollowTable_Benchmark/data/$DS/query_instruction.json \
#         --out_json     irs_$DS.json
# done

# # python notify.py



# ################

# MODEL=deu05232/promptriever-llama2-7B-add_all_q

# MODEL_SAFE=${MODEL//\//_}


# for DS in WQT WTR TArX IndusTR
# do
#     echo "Processing dataset: $DS"
#     DATA=/workspace/FollowTable_Benchmark/data/FollowTable/$DS

#     # CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
#     #     --corpus_file corpus.jsonl --query_file queries.jsonl --split test \
#     #     --model_path $MODEL --per_gpu_batch_size 2 --order "query_first_eos_token"

#     # CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
#     #     --corpus_file corpus.jsonl --query_file only_queries.jsonl --split for_only_query_test \
#     #     --model_path $MODEL --per_gpu_batch_size 2 --order "only_query"

#     CUDA_VISIBLE_DEVICES=0 python irs.py \
#         --model_name $MODEL_SAFE \
#         --instr_pickle model_pred/$DS/corpus/test/queries/query_first_eos_token/$MODEL_SAFE.pickle \
#         --base_pickle  model_pred/$DS/corpus/for_only_query_test/only_queries/only_query/$MODEL_SAFE.pickle \
#         --raw_json     /workspace/FollowTable_Benchmark/data/$DS/query_instruction.json \
#         --out_json     irs_$DS.json
# done

# python notify.py



MODEL=deu05232/repllama-llama2-7B-followtable

MODEL_SAFE=${MODEL//\//_}


for DS in WQT WTR TArX IndusTR
do
    echo "Processing dataset: $DS"
    DATA=/workspace/FollowTable_Benchmark/data/FollowTable/$DS

    CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
        --corpus_file corpus.jsonl --query_file queries.jsonl --split test \
        --model_path $MODEL --per_gpu_batch_size 4 --order "query_first_eos_token"

    CUDA_VISIBLE_DEVICES=0 python eval_instructir_dense.py --eval_datasets_dir $DATA \
        --corpus_file corpus.jsonl --query_file only_queries.jsonl --split for_only_query_test \
        --model_path $MODEL --per_gpu_batch_size 4 --order "only_query"

    CUDA_VISIBLE_DEVICES=0 python irs.py \
        --model_name $MODEL_SAFE \
        --instr_pickle model_pred/$DS/corpus/test/queries/query_first_eos_token/$MODEL_SAFE.pickle \
        --base_pickle  model_pred/$DS/corpus/for_only_query_test/only_queries/only_query/$MODEL_SAFE.pickle \
        --raw_json     /workspace/FollowTable_Benchmark/data/$DS/query_instruction.json \
        --out_json     irs_$DS.json
done

python notify.py


# # # irs 확인용
# # CUDA_VISIBLE_DEVICES=0 python irs.py \
# #     --instr_pickle /workspace/FollowTable_Benchmark/eval/model_pred/FollowTable/corpus/test/queries/query_first_eos_token/Qwen_Qwen3-Embedding-8B.pickle \
# #     --base_pickle  /workspace/FollowTable_Benchmark/eval/model_pred/FollowTable/corpus/for_only_query_test/only_queries/only_query/Qwen_Qwen3-Embedding-8B.pickle \
# #     --raw_json     /workspace/FollowTable_Benchmark/data/WQT/query_instruction.json \
# #     --out_json     irs_WQT.json
