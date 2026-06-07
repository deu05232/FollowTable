'''
We reformulate the code from TART (https://github.com/facebookresearch/tart)
'''
import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from tqdm import tqdm
import glob

from . import dist_utils as dist_utils
from . import normalize_text

class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        prompt=None,
        prompt_only=False,
        emb_load_path=None,
        emb_save_path=None,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.prompt = prompt
        self.prompt_only = prompt_only
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        if self.prompt is not None:
            if self.prompt_only:
                queries = ["{1} [SEP] {0}".format(self.prompt, query) for query in queries]
            else:
                queries = ["{0} [SEP] {1}".format(self.prompt, query) for query in queries]
            print(queries[-1])

        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.to(self.query_encoder.device) for key, value in qencode.items()}
                emb = self.query_encoder(**qencode, normalize=self.norm_query)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        
        if self.emb_load_path is None:
            corpus = [corpus[i] for i in idx]
            corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
            if self.normalize_text:
                corpus = [normalize_text.normalize(c) for c in corpus]
            if self.lower_case:
                corpus = [c.lower() for c in corpus]

            allemb = []
            nbatch = (len(corpus) - 1) // batch_size + 1
            with torch.no_grad():
                for k in tqdm(range(nbatch)):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus))

                    cencode = self.tokenizer.batch_encode_plus(
                        corpus[start_idx:end_idx],
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                        return_tensors="pt",
                    )
                    cencode = {key: value.to(self.doc_encoder.device) for key, value in cencode.items()}
                    emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                    allemb.append(emb.cpu())

            allemb = torch.cat(allemb, dim=0)
            if self.emb_save_path is not None:
                torch.save(allemb, self.emb_save_path)
        else:
            print("loading from {}".format(self.emb_load_path))
            embs_list = []
            for path in self.emb_load_path:
                embs = torch.load(path)
                print(embs.size())
                embs_list.append(embs)
                print(len(embs_list))
                
            allemb = torch.cat(embs_list, dim=0)
            print(allemb.size())
            
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb


class DenseEncoder_w_Decoder_Model:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        prompt=None,
        prompt_only=False,
        emb_load_path=None,
        emb_save_path=None,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.prompt = prompt
        self.prompt_only = prompt_only
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] + "</s>" for i in idx]
        if self.prompt is not None:
            if self.prompt_only:
                queries = ["query: {1} [SEP] {0}</s>".format(self.prompt, query) for query in queries]
            else:
                queries = ["{0} [SEP] query: {1}".format(self.prompt, query) for query in queries]
            print(queries[-1])

        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.to(self.query_encoder.device) for key, value in qencode.items()}
                q_outputs = self.query_encoder(**qencode)
                query_embedding = q_outputs.last_hidden_state[:,-1,:]
                emb = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        print("all qemb:",allemb.shape)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        
        if self.emb_load_path is None:
            corpus = [corpus[i] for i in idx]
            corpus = [c["title"] + " " + c["text"] + "</s>" if len(c["title"]) > 0 else c["text"] + "</s>" for c in corpus]
            if self.normalize_text:
                corpus = [normalize_text.normalize(c) for c in corpus]
            if self.lower_case:
                corpus = [c.lower() for c in corpus]

            allemb = []
            nbatch = (len(corpus) - 1) // batch_size + 1
            with torch.no_grad():
                for k in tqdm(range(nbatch)):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus))

                    cencode = self.tokenizer.batch_encode_plus(
                        corpus[start_idx:end_idx],
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                        return_tensors="pt",
                    )
                    cencode = {key: value.to(self.doc_encoder.device) for key, value in cencode.items()}
                    passage_outputs = self.doc_encoder(**cencode)
                    passage_embeddings = passage_outputs.last_hidden_state[:,-1,:]
                    emb = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1) ## change! FIX!!
                    allemb.append(emb.cpu())

            allemb = torch.cat(allemb, dim=0)
            print("all cemb:",allemb.shape)
            if self.emb_save_path is not None:
                torch.save(allemb, self.emb_save_path)
        else:
            print("loading from {}".format(self.emb_load_path))
            embs_list = []
            for path in self.emb_load_path:
                embs = torch.load(path)
                print(embs.size())
                embs_list.append(embs)
                print(len(embs_list))
                
            allemb = torch.cat(embs_list, dim=0)
            print(allemb.size())
            
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

class DenseEncoder_w_promptriever:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        query_prompt="query:",
        passage_prompt="passage:",
        seperator=None,
        emb_load_path=None,
        emb_save_path=None,
        order="query_first",
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt
        self.seperator = seperator
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path
        self.order = order
        
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))
            
        if self.order == "only_query":
            queries = [queries[i] for i in idx]
        elif "instruction_first" in self.order:
            if self.seperator:
                queries = [queries[i].replace("[SEP]", self.seperator) for i in idx]
            else:
                queries = [queries[i].replace("[SEP] ", "") for i in idx]
        elif "query_first" in self.order:
            tmp = []
            for i in idx:
                query = queries[i]
                inst, q = query.split(" [SEP] ")
                if self.seperator:
                    tmp.append(f"{q}? {self.seperator} {inst}")
                else:
                    tmp.append(f"{q}? {inst}")
            queries = tmp
        elif self.order == "special_token":
            queries = [queries[i].replace("[SEP] ", "") for i in idx]
        elif self.order == "special_token_sep":
            if self.seperator == "eos_token":
                self.seperator = self.tokenizer.eos_token
            queries = [queries[i].replace("[SEP] ", self.seperator) for i in idx]
        elif self.order == "instruction_first_split":
            queries = ["instruction: " + queries[i].replace("[SEP]", "query: ") for i in idx]
        else:
            assert False, "hm"
            
        if self.query_prompt is not None:
            if self.order == "instruction_first_split":
                pass
            else:
                queries = [f"{self.query_prompt} {q}" for q in queries]

        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        print("query: ", queries[0])

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                # Tokenize with max_length - 1 to leave room for EOS token
                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length - 1,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    padding=False,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                )

                # Manually add EOS token at the end
                qencode["input_ids"] = [
                    input_ids + [self.tokenizer.eos_token_id]
                    for input_ids in qencode["input_ids"]
                ]

                # Pad the batch
                qencode = self.tokenizer.pad(
                    qencode,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                qencode = {key: value.to(self.query_encoder.device) for key, value in qencode.items()}
                q_outputs = self.query_encoder(**qencode)
                last_hidden_state = q_outputs.last_hidden_state
                sequence_lengths = qencode["attention_mask"].sum(dim=1) - 1
                batch_size_actual = last_hidden_state.shape[0]
                reps = last_hidden_state[
                        torch.arange(batch_size_actual, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                emb = torch.nn.functional.normalize(reps, p=2, dim=1)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        print("all qemb:",allemb.shape)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        
        if self.emb_load_path is None:
            corpus = [corpus[i] for i in idx]
            if self.passage_prompt is not None:
                corpus = [self.passage_prompt + " " + c["title"] + " " + c["text"] if len(c["title"]) > 0 else self.passage_prompt + " " + c["text"] for c in corpus]
            else:
                corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]

            if self.normalize_text:
                corpus = [normalize_text.normalize(c) for c in corpus]
            if self.lower_case:
                corpus = [c.lower() for c in corpus]

            allemb = []
            nbatch = (len(corpus) - 1) // batch_size + 1
            with torch.no_grad():
                for k in tqdm(range(nbatch)):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus))

                    # Tokenize with max_length - 1 to leave room for EOS token
                    cencode = self.tokenizer.batch_encode_plus(
                        corpus[start_idx:end_idx],
                        max_length=self.max_length - 1,
                        return_token_type_ids=False,
                        return_attention_mask=False,
                        padding=False,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                    )

                    # Manually add EOS token at the end
                    cencode["input_ids"] = [
                        input_ids + [self.tokenizer.eos_token_id]
                        for input_ids in cencode["input_ids"]
                    ]

                    # Pad the batch
                    cencode = self.tokenizer.pad(
                        cencode,
                        padding=True,
                        pad_to_multiple_of=8,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )

                    cencode = {key: value.to(self.doc_encoder.device) for key, value in cencode.items()}
                    passage_outputs = self.doc_encoder(**cencode)
                    last_hidden_state = passage_outputs.last_hidden_state
                    sequence_lengths = cencode["attention_mask"].sum(dim=1) - 1
                    batch_size_actual = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                            torch.arange(batch_size_actual, device=last_hidden_state.device),
                            sequence_lengths,
                        ]
                    emb = torch.nn.functional.normalize(reps, p=2, dim=1)
                    allemb.append(emb.cpu())

            allemb = torch.cat(allemb, dim=0)
            print("all cemb:",allemb.shape)
            if self.emb_save_path is not None:
                torch.save(allemb, self.emb_save_path)
        else:
            print("loading from {}".format(self.emb_load_path))
            embs_list = []
            for path in self.emb_load_path:
                embs = torch.load(path)
                print(embs.size())
                embs_list.append(embs)
                print(len(embs_list))
                
            allemb = torch.cat(embs_list, dim=0)
            print(allemb.size())
            
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb
    
    
class DenseEncoder_w_mistral:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        prompt=None,
        prompt_only=False,
        emb_load_path=None,
        emb_save_path=None,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.prompt = prompt
        self.prompt_only = prompt_only
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path

    def last_token_pool(self,last_hidden_states: Tensor,attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [f"Instruct: {queries[i].split('[SEP]')[0].strip()}\nQuery: {queries[i].split('[SEP]')[1].strip()}" + "</s>" for i in idx]
        if self.prompt is not None:
            if self.prompt_only:
                queries = ["Query: {1} {0}".format(self.prompt, query)+ "</s>"  for query in queries]
            else:
                queries = ["Instruct: {0}\nQuery: {1}".format(self.prompt, query)+ "</s>" for query in queries]
            print(queries[-1])

        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length -1,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.to(self.query_encoder.device) for key, value in qencode.items()}
                q_outputs = self.query_encoder(**qencode)

                query_embedding = self.last_token_pool(q_outputs.last_hidden_state, qencode['attention_mask'])
                emb = torch.nn.functional.normalize(query_embedding, p=2, dim=1) # row normalize

                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        print("all qemb:",allemb.shape)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        
        if self.emb_load_path is None:
            corpus = [corpus[i] for i in idx]
            corpus = [c["title"] + " " + c["text"] + "</s>" if len(c["title"]) > 0 else c["text"] + "</s>" for c in corpus]
            if self.normalize_text:
                corpus = [normalize_text.normalize(c) for c in corpus]
            if self.lower_case:
                corpus = [c.lower() for c in corpus]

            allemb = []
            nbatch = (len(corpus) - 1) // batch_size + 1
            with torch.no_grad():
                for k in tqdm(range(nbatch)):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus))

                    cencode = self.tokenizer.batch_encode_plus(
                        corpus[start_idx:end_idx],
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                        return_tensors="pt",
                    )
                    cencode = {key: value.to(self.doc_encoder.device) for key, value in cencode.items()}
                    passage_outputs = self.doc_encoder(**cencode)

                    passage_embeddings = self.last_token_pool(passage_outputs.last_hidden_state, cencode['attention_mask'])
                    emb = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1)
                    allemb.append(emb.cpu())

            allemb = torch.cat(allemb, dim=0)
            print("all cemb:",allemb.shape)
            if self.emb_save_path is not None:
                torch.save(allemb, self.emb_save_path)
        else:
            print("loading from {}".format(self.emb_load_path))
            embs_list = []
            for path in self.emb_load_path:
                embs = torch.load(path)
                print(embs.size())
                embs_list.append(embs)
                print(len(embs_list))
                
            allemb = torch.cat(embs_list, dim=0)
            print(allemb.size())
            
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb


class DenseEncoderModelInstructor:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        prompt=None,
        corpus_prompt=None, # hs 
        prompt_only = False,
        emb_load_path=None,
        emb_save_path=None,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        # self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.prompt = prompt
        self.corpus_prompt = corpus_prompt # hs
        self.prompt_only = prompt_only
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path

        print("### self.prompt:", self.prompt)
        print("### self.corpus_prompt:", self.corpus_prompt)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        if isinstance(queries[0],str):
            if not '[SEP]' in queries[0]:
                queries = [[None, queries[i]] for i in idx]
            else:
                queries = [queries[i].split('[SEP]') for i in idx]
                
        elif isinstance(queries[0],list):
            queries = [[queries[i][0],queries[i][1]] for i in idx]

        if self.prompt is not None:
            if self.prompt_only:
                queries = [ [query[1],self.prompt ] for query in queries]
            else:
                queries = [ [self.prompt, query[1] ] for query in queries]
            print(queries[-1])

        if self.normalize_text:
            queries = [[normalize_text.normalize(q[0]),normalize_text.normalize(q[1])] for q in queries]
        if self.lower_case:
            queries = [[q[0].lower(),q[1].lower() ] for q in queries]

        print(queries[-1])
        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                emb = self.query_encoder.encode(queries[start_idx:end_idx], normalize_embeddings= self.norm_query,  convert_to_tensor=True)
                allemb.append(emb.cpu())
                
        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        
        if self.emb_load_path is None:
            corpus = [[None,  c["title"] + " " + c["text"]] if len(c["title"]) > 0 else [None, c["text"]] for c in corpus]

            if self.corpus_prompt is not None: # hs
                corpus = [[self.corpus_prompt, c[1]] for c in corpus]
                print(corpus[-1])

            if self.normalize_text:
                corpus = [[normalize_text.normalize(c[0]), normalize_text.normalize(c[1])] for c in corpus]
            if self.lower_case:
                corpus = [[c[0].lower(),c[1].lower()] for c in corpus]

            allemb = []
            nbatch = (len(corpus) - 1) // batch_size + 1
            with torch.no_grad():
                for k in tqdm(range(nbatch)):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus))
                    emb = self.doc_encoder.encode(corpus[start_idx:end_idx], normalize_embeddings= self.norm_doc,convert_to_tensor=True)
                    allemb.append(emb.cpu())

            allemb = torch.cat(allemb, dim=0)
            if self.emb_save_path is not None:
                torch.save(allemb, self.emb_save_path)
        else:
            print("loading from {}".format(self.emb_load_path))
            embs_list = []
            for path in self.emb_load_path:
                embs = torch.load(path)
                print(embs.size())
                embs_list.append(embs)
                print(len(embs_list))
                
            allemb = torch.cat(embs_list, dim=0)
            print(allemb.size())
            
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

def evaluate_model(
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    prompt=None,
    emb_load_path=None,
    emb_save_path=None,
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
            prompt=prompt,
            emb_load_path=emb_load_path,
            emb_save_path=emb_save_path,
        ),
        batch_size=batch_size,
    )
    retriever = EvaluateRetrieval(dmodel, score_function=score_function)
    data_path = os.path.join(beir_dir, dataset)

    if not os.path.isdir(data_path) and is_main:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    dist_utils.barrier()

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        results = retriever.retrieve(corpus, queries)
        if is_main:
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
            if save_results_path is not None:
                torch.save(results, f"{save_results_path}")
    elif dataset == "cqadupstack":  # compute macroaverage over datasets
        paths = glob.glob(data_path)
        for path in paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
            results = retriever.retrieve(corpus, queries)
            if is_main:
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                    if isinstance(metric, str):
                        metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                    for key, value in metric.items():
                        metrics[key].append(value)
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics


class DenseEncoder_nvemb:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None, 
        tokenizer=None,
        max_length=512,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        query_prompt="Instruct: Given a question, retrieve passages that answer the question.\nQuery: ",
        passage_prompt="",
        seperator=None,
        emb_load_path=None,
        emb_save_path=None,
        order="query_first",
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt
        self.seperator = seperator
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path
        self.order = order
        
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))
        
        if self.order == "query_first":
            tmp = []
            for query in queries:
                inst, q = query.split(" [SEP] ")
                if self.seperator:
                    tmp.append(f"{q} {self.seperator} {inst}</s>")
                else:
                    tmp.append(f"{q} {inst}</s>")
            queries = tmp
        elif self.order == "instruction_first":
            if self.seperator:
                queries = [queries[i].replace("[SEP]", self.seperator) + "</s>" for i in idx]
            else:
                queries = [queries[i].replace("[SEP] ", "") + "</s>" for i in idx]
        elif self.order == "query_first_eos_token":
            tmp = []
            eos_token = self.tokenizer.eos_token if self.tokenizer is not None else "</s>"
            for query in queries:
                inst, q = query.split(" [SEP] ")
                if self.seperator:
                    tmp.append(f"{q} {eos_token} {inst}{eos_token}")
                else:
                    tmp.append(f"{q} {inst}{eos_token}")
            queries = tmp
        elif self.order == "special_token":
            queries = [queries[i].replace("[SEP] ", "") + self.seperator for i in idx]
        elif self.order == "special_token_sep":
            queries = [queries[i].replace("[SEP] ", self.seperator) + self.seperator for i in idx]
        elif self.order == "instruction_first_split":
            queries = ["instruction: " + queries[i].replace("[SEP]", "query: ") + "</s>" for i in idx]
        else:
            assert False, "hm"
            
        if self.query_prompt is not None:
            if self.order == "instruction_first_split":
                pass
            else:
                queries = [f"{self.query_prompt} {q}" for q in queries]

        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        embeddings = self.query_encoder._do_encode(
            queries,
            batch_size=batch_size,
            instruction=None,  # Already prefixed
            max_length=self.max_length,
            num_workers=4,
            return_numpy=False
        )

        if self.norm_query:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        
        if self.emb_load_path is None:
            corpus = [corpus[i] for i in idx]
            if self.passage_prompt is not None:
                corpus = [self.passage_prompt + " " + c["title"] + " " + c["text"] + "</s>" if len(c["title"]) > 0 else c["text"] + "</s>" for c in corpus]
            else:
                corpus = [c["title"] + " " + c["text"] + "</s>" if len(c["title"]) > 0 else c["text"] + "</s>" for c in corpus]
                
            if self.normalize_text:
                corpus = [normalize_text.normalize(c) for c in corpus]
            if self.lower_case:
                corpus = [c.lower() for c in corpus]

            embeddings = self.doc_encoder._do_encode(
                corpus,
                batch_size=batch_size,
                instruction=None,
                max_length=self.max_length,
                num_workers=4,
                return_numpy=False
            )

            if self.norm_doc:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            if self.emb_save_path is not None:
                torch.save(embeddings, self.emb_save_path)
                
            return embeddings.cpu().numpy()
        # else:
        #     print("loading from {}".format(self.emb_load_path))
        #     embs_list = []
        #     for path in self.emb_load_path:
        #         embs = torch.load(path)
        #         print(embs.size())
        #         embs_list.append(embs)
        #         print(len(embs_list))
                
        #     allemb = torch.cat(embs_list, dim=0)
        #     print(allemb.size())
            
        # allemb = allemb.cuda()
        # if dist.is_initialized():
        #     allemb = dist_utils.varsize_gather_nograd(allemb)
        # allemb = allemb.cpu().numpy()
        # return allemb


class DenseEncoder_w_qwen3:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None, 
        max_length=512,
        query_prompt="",
        emb_load_path=None,
        emb_save_path=None,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder if doc_encoder else query_encoder
        self.max_length = max_length
        self.normalize_text = normalize_text
        self.query_prompt = query_prompt
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path
        
        
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        """
        Encode queries using the query encoder with query prompt
        """
        if self.query_prompt:
            embeddings = self.query_encoder.encode(
                queries, 
                batch_size=batch_size,
                prompt=self.query_prompt,
                # convert_to_tensor=True,
                **kwargs
            )
        else:
            # Encode queries with prompt_name="query" as shown in Qwen3 example
            embeddings = self.query_encoder.encode(
                queries, 
                batch_size=batch_size,
                prompt_name="query",
                # convert_to_tensor=True,
                **kwargs
            )
        
        return embeddings.cpu().numpy()


    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        """
        Encode corpus documents
        """
        # Extract text from corpus dictionaries
        # Assuming corpus contains documents with 'title' and 'text' fields
        documents = []
        for doc in corpus:
            if isinstance(doc, dict):
                # Combine title and text if both exist
                title = doc.get('title', '')
                text = doc.get('text', '')
            
                if title and text:
                    combined_text = f"{title} {text}"
                elif text:
                    combined_text = text
                elif title:
                    combined_text = title
                else:
                    combined_text = str(doc)
                    
                documents.append(combined_text)
            else:
                # If corpus item is not a dict, convert to string
                documents.append(doc)
        
        # Encode documents (no query prompt for documents)
        embeddings = self.doc_encoder.encode(
            documents,
            batch_size=batch_size,
            # convert_to_tensor=True,
            **kwargs
        )
    
        return embeddings.cpu().numpy()
    
    
class DenseEncoder_grit:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        max_length=512,
        norm_query=False,
        norm_doc=False,
        query_prompt="Given a question, retrieve passages that answer the question.",
        passage_prompt="",
        seperator=None,
        emb_load_path=None,
        emb_save_path=None,
        order='only_query',
        **kwargs,
    ):

        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.max_length = max_length
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt
        self.seperator = seperator
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path
        self.order = order

        # 임베딩이 저장된 경로가 있다면 로드
        if self.emb_load_path is not None:
            try:
                self.cached_embeddings = torch.load(self.emb_load_path)
                print(f"Loaded embeddings from {self.emb_load_path}")
            except:
                self.cached_embeddings = None
                print(f"Failed to load embeddings from {self.emb_load_path}")
        else:
            self.cached_embeddings = None
    
    def gritlm_instruction(self, instruction: str) -> str:
        """GritLM instruction 포맷팅"""
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    
    
    def encode_queries(self, queries: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """쿼리들을 인코딩하여 임베딩 반환"""
        if self.order == "only_query":
            instruction = self.gritlm_instruction(self.query_prompt)
            
            # 배치 단위로 인코딩
            all_embeddings = []
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                
                # GritLM 모델로 인코딩
                batch_embeddings = self.query_encoder.encode(
                    batch_queries, 
                    instruction=instruction,
                    max_length=self.max_length,
                    **kwargs
                )
                
                # 정규화 적용
                if self.norm_query:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                all_embeddings.append(batch_embeddings)

        else:
            print("기존 방식대로 들어감")

            qs, instructions = [], []
            for query in queries:
                inst, q = query.split(" [SEP] ")
                qs.append(q.strip())
                instructions.append(inst.strip())

            queries = qs

            # GritLM instruction 적용
            instructions = [self.gritlm_instruction(inst) for inst in instructions]
            
            # 배치 단위로 인코딩
            all_embeddings = []
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_instructions = instructions[i:i + batch_size]
                
                # GritLM 모델로 인코딩
                batch_embeddings = self.query_encoder.encode(
                    batch_queries, 
                    instructions=batch_instructions,
                    max_length=self.max_length,
                    **kwargs
                )
                
                # 정규화 적용
                if self.norm_query:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                all_embeddings.append(batch_embeddings)
        
        # 모든 배치 결합
        embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings.cpu().numpy()

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        """코퍼스(문서들)를 인코딩하여 임베딩 반환"""
        
        # 캐시된 임베딩이 있다면 반환
        if self.cached_embeddings is not None:
            print("Using cached embeddings")
            return self.cached_embeddings.cpu().numpy()
        
        # 문서 텍스트 추출 및 전처리
        if isinstance(corpus[0], dict):
            # corpus가 딕셔너리 리스트인 경우 (title, text 등의 키를 가짐)
            documents = []
            for doc in corpus:
                if 'title' in doc and 'text' in doc:
                    if self.seperator:
                        doc_text = doc['title'] + self.seperator + doc['text']
                    else:
                        doc_text = doc['title'] + " " + doc['text']
                elif 'text' in doc:
                    doc_text = doc['text']
                elif 'title' in doc:
                    doc_text = doc['title']
                else:
                    # 첫 번째 값을 사용
                    doc_text = str(list(doc.values())[0])
                
                documents.append(doc_text)
        else:
            # corpus가 문자열 리스트인 경우
            documents = [doc for doc in corpus]
        
        
        # GritLM instruction (문서의 경우 빈 instruction 사용)
        instruction = self.gritlm_instruction("")
        
        # 배치 단위로 인코딩
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            print(f"Encoding batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # GritLM 모델로 인코딩
            batch_embeddings = self.doc_encoder.encode(
                batch_docs,
                instruction=instruction,
                max_length=self.max_length,
                **kwargs
            )
            
            # 정규화 적용
            if self.norm_doc:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            all_embeddings.append(batch_embeddings)
        
        # 모든 배치 결합
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # 임베딩 저장
        if self.emb_save_path is not None:
            torch.save(embeddings, self.emb_save_path)
            print(f"Saved embeddings to {self.emb_save_path}")
        
        return embeddings.cpu().numpy()