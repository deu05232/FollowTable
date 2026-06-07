from transformers import AutoModel, AutoTokenizer
import logging
import os
import random
import torch 
import argparse
import pickle 
from peft import PeftModel, PeftConfig
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir import util, LoggingHandler
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer


from src.beir_utils import (
    DenseEncoderModel,
    DenseEncoderModelInstructor, 
    DenseEncoder_w_Decoder_Model,
    DenseEncoder_w_promptriever,
    DenseEncoder_w_mistral,
    DenseEncoder_w_qwen3,
    DenseEncoder_grit,
    DenseEncoder_nvemb)
from src import peft_utils
from src.options import Arguments
from robust_eval import CustomEvaluateRetrieval


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def load_model(args, new_args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _prompt = new_args.prompt
    _prompt_only = new_args.prompt_only
    
    def _get_model(peft_model_name):
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
        print("base_model: ", config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        model.eval()
        return model

    def _get_tokenizer(peft_model_name):
        config = PeftConfig.from_pretrained(peft_model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        return tokenizer
    
    # if ('ckpt' in args.model_path) or ('contriever' in args.model_path):
    if ('contriever' in args.model_path):
        print("# Contriever / TART-dual")
        retriever, peft_config, tokenizer = peft_utils.create_and_prepare_model(args)
        retriever.config.use_cache = False

        retriever = retriever.to(device)

        dmodel = DRES(
            DenseEncoderModel(
                query_encoder=retriever,
                doc_encoder=retriever,
                tokenizer=tokenizer,
                prompt = _prompt, # When using TART you should specify prompt
                prompt_only=_prompt_only # ONLY Activate when only using instruction as a query
            ),
            batch_size=args.per_gpu_batch_size, 
        )
        _score_function = "dot"

    elif 'qwen3' in args.model_path.lower():
        print("# Qwen-3-7B")
        retriever = SentenceTransformer(model_name_or_path="Qwen/Qwen3-Embedding-8B",
                                        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
                                        tokenizer_kwargs={"padding_side": "left"}
                                        )
    

        retriever = retriever.to(device)
        
        dmodel = DRES(
            DenseEncoder_w_qwen3(
                query_encoder=retriever,
                doc_encoder=retriever,
                query_prompt = "", # 없으면 default값으로 들어감
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "dot"
        

    elif 'hkunlp' in args.model_path: 
        print("# INSTRUCTOR variants") 
        retriever = INSTRUCTOR(args.model_path) 
        retriever = retriever.to(device)
        
        dmodel = DRES(
            DenseEncoderModelInstructor(
                query_encoder=retriever,
                doc_encoder=retriever,
                prompt = _prompt, 
                corpus_prompt = 'Represent the document for retrieval:',
                prompt_only=_prompt_only # ONLY Activate when only using instruction as a query
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "cos_sim"
        
    elif 'promptriever' in args.model_path.lower() or 'repllama' in args.model_path.lower() or 'revela' in args.model_path.lower() :
    # elif 'checkpoint' in args.model_path.lower() and ('promptriever' in args.model_path.lower() or 'llama' in args.model_path.lower()  or 'qwen' in args.model_path.lower()):
        print("*************** custom promptriever")
        tokenizer = _get_tokenizer(args.model_path)
        # print("tokenizer: ", tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 512
        tokenizer.padding_side = "right"
        retriever = _get_model(args.model_path)
        retriever.config.max_length = 512
        retriever = retriever.to(device)

        if "-instruct" in args.model_path.lower() and "no_template" not in args.model_path.lower():
            # llama_format = "<|start_header_id|>user<|end_header_id|>\n\n"
            llama_format = "<|start_header_id|>system<|end_header_id|>\n\nYou are an embedding model. Return a representation for semantic search.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            query_prompt = llama_format + "query:"
            passage_prompt = llama_format + "passage:"
        else:
            query_prompt = "query:"
            passage_prompt = "passage:"
        
        if new_args.seperator is not None:
            print(f"***seperator 사용: {new_args.seperator}***")

        
        dmodel = DRES(
            DenseEncoder_w_promptriever(
                query_encoder=retriever,
                doc_encoder=retriever,
                tokenizer=tokenizer,
                query_prompt = query_prompt, # When using TART you should specify prompt
                passage_prompt = passage_prompt,
                seperator = new_args.seperator,
                order=new_args.order
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "dot"
      
    elif 'mistral' in args.model_path:
        print("# E5-mistral-7b-instruct")
        
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        retriever = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')

        tokenizer.pad_token = tokenizer.eos_token
        retriever = retriever.to(device)

        dmodel = DRES(
            DenseEncoder_w_mistral(
                query_encoder=retriever,
                doc_encoder=retriever,
                tokenizer=tokenizer,
                prompt = _prompt, # When using TART you should specify prompt
                prompt_only=_prompt_only # ONLY Activate when only using instruction as a query
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "dot"
      
    elif 'nv-embed' in args.model_path.lower(): 
        print("# NV-Embed")
        
        tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
        retriever = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)

        tokenizer.pad_token = tokenizer.eos_token
        retriever = retriever.to(device)
        tokenizer.model_max_length = 512
        
        # Instruct: Given a question, retrieve passages that answer the question.\n
        dmodel = DRES(
            DenseEncoder_nvemb(
                query_encoder=retriever,
                doc_encoder=retriever,
                tokenizer=tokenizer,
                query_prompt = "Query:", # When using TART you should specify prompt
                passage_prompt = "",
                order=new_args.order
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "dot"
        
    elif "GTR" in args.model_path:
        print("# GTR variants")
        dmodel = models.SentenceBERT(args.model_path)
        _score_function = "cos_sim"
        dmodel = DRES(
            dmodel, 
            batch_size=args.per_gpu_batch_size, #128,
        )
    elif 'grit' in args.model_path.lower(): 
        from gritlm import GritLM

        print("# gritlm")
        
        def get_gpus_max_memory(max_memory):
            max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
            return max_memory

        retriever = GritLM("GritLM/GritLM-7B", 
                            torch_dtype=torch.bfloat16,
                            normalized=False,
                            mode="embedding",
                            pooling_method="mean",
                            attn_implementation="sdpa",
                            attn='bbcc',
                            device_map="auto",
                            max_memory=get_gpus_max_memory("48GB"),
                            offload_folder="offload",
                            )
        
        retriever.order = new_args.order

        # Instruct: Given a question, retrieve passages that answer the question.\n
        dmodel = DRES(
            DenseEncoder_grit(
                query_encoder=retriever,
                doc_encoder=retriever,
                query_prompt = "Given a question, retrieve passages that answer the question.", # When using TART you should specify prompt
                passage_prompt = "",
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "cos_sim"
    #     
    else:
        assert False, "Model not supported yet."

    return dmodel, _score_function

def main(args):
    arguments = Arguments()
    og_args = arguments.parse()

    dataset_name = og_args.eval_datasets_dir.split('/')[-1]

    logging.info(f"OG_Args:\n{og_args}\nNew Args:\n{args}")
    
    corpus, queries, qrels = GenericDataLoader(
        data_folder=og_args.eval_datasets_dir,
        corpus_file=args.corpus_file,
        qrels_file= os.path.join(og_args.eval_datasets_dir, args.qrels_folder, args.split + '.tsv'),
        query_file=args.query_file,
        ).load_custom()

    logging.info(f"len(corpus),len(queries),len(qrels): {len(corpus),len(queries),len(qrels)}")

    dmodel, _score_function = load_model(og_args, args)
    
    # retriever = EvaluateRetrieval(dmodel, score_function=_score_function)
    retriever = CustomEvaluateRetrieval(dmodel, score_function=_score_function)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    if args.order == "only_query":
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        ndcg10 = ndcg['NDCG@10']
        with open("metrics_results_only_query.txt", "a+", encoding="utf-8") as f:
            f.write(f"===== {og_args.model_path} - {dataset_name} =====\n")
            f.write(f"nDCG@10: {ndcg10:.4f}\n")
            f.write("\n")
            
    

    else:
        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        metrics = retriever.robustness_evaluate(queries, qrels, results, retriever.k_values, type='query')
        
        nDCG10_avg = metrics['cluster_NDCG@10_avg']
        nDCG10_min_avg = metrics['cluster_NDCG@10_min_avg']
        nDCG1_avg = metrics['cluster_NDCG@1_avg']
        nDCG1_min_avg = metrics['cluster_NDCG@1_min_avg']
        nDCG5_avg = metrics['cluster_NDCG@5_avg']
        nDCG5_min_avg = metrics['cluster_NDCG@5_min_avg']
        
        with open("metrics_results.txt", "a+", encoding="utf-8") as f:
            f.write(f"===== {og_args.model_path} - {dataset_name} =====\n")
            f.write(f"nDCG@10 avg: {nDCG10_avg:.4f}\n")
            f.write(f"nDCG@10 min avg: {nDCG10_min_avg:.4f}\n")
            f.write(f"nDCG@1 avg: {nDCG1_avg:.4f}\n")
            f.write(f"nDCG@1 min avg: {nDCG1_min_avg:.4f}\n")
            f.write(f"nDCG@5 avg: {nDCG5_avg:.4f}\n")
            f.write(f"nDCG@5 min avg: {nDCG5_min_avg:.4f}\n")
            f.write("\n")

    #### Save
    data_path = og_args.eval_datasets_dir.split('/')[-2]
    corpus_file = args.corpus_file.split('.')[0]
    split_flag = args.split
    query_version = args.query_file.split('.')[0].replace('/','_')

    save_path = f'model_pred/{dataset_name}/{corpus_file}/{split_flag}/{query_version}/{args.order}/'

    os.makedirs(save_path, exist_ok=True)
    save_path += og_args.model_path.replace('..','_').replace('/','_') + '.pickle'
    with open(save_path,'wb') as f:
        pickle.dump(results,f)
        

if __name__=="__main__":    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--split", type=str,default='test',
                        help="split for qrels")
    
    parser.add_argument("--corpus_file", type=str, default='corpus.jsonl',
                        help="corpus file name")
    parser.add_argument("--query_file", type=str, default='queries.jsonl',
                        help="query file name")
    parser.add_argument("--qrels_folder", type=str, default='qrels',
                        help="qrels folder name")
    
    parser.add_argument(
        "--prompt", type=str, default=None, help="instructional prompt."
    )
    parser.add_argument(
        "--prompt_only", action="store_true", help="only use prompt"
    )
    
    parser.add_argument(
        "--order", type=str, default=None, help="instructional prompt."
    )
    
    parser.add_argument(
        "--seperator", type=str, default=None, help="instructional prompt."
    )
    
    # query_prompt = "query:"
    # passage_prompt = "passage:"
    
    cfg, _ = parser.parse_known_args()

    main(cfg)

