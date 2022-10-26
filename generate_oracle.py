from constants import BOS_TOKEN, EOT_TOKEN, PAD_TOKEN, SEPARATOR, CLASSIFIER_DROPOUT
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer
# from predict_alignment import predict_alignment
from model import OracleBinaryClassifier, OracleClassifier
from constants import LANGUAGE_MODEL_STRING
from util import num_params
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=int, default=None)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--output', type=str, required=True)

    parser.add_argument('--no_label', action='store_true', help='only consider commands without label')
    parser.add_argument('--no_length', action='store_true', help='only consider commands without length')

    # special
    parser.add_argument('--multiclass', action='store_true', help='use OracleClassifier instead of OracleBinaryClassifier')

    # models
    parser.add_argument('--ckpt_length', type=str, default=None, help='leave as None to control length only')
    parser.add_argument('--ckpt_label', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--label_num_classes', type=int, default=5)
    parser.add_argument('--length_num_classes', type=int, default=5)

    # decode config
    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--condition_lambda_length', type=float, default=1.0, help='lambda weight on length conditioning model')
    parser.add_argument('--condition_lambda_label', type=float, default=1.0, help='lambda weight on label conditioning model')    
    parser.add_argument('--length_cutoff', type=int, default=256, help='max length')
    parser.add_argument('--min_length', type=int, default=0, help='min length')
    parser.add_argument('--topk', type=int, default=50, help='top k for sampling')

    # misc
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--no_condition_past', action='store_true')
    

    return parser.parse_args()

def load_oracle_model_from_ckpt(args, device, ckpt, tokenizer, n_classes):
    ModelClass = OracleClassifier if args.multiclass else OracleBinaryClassifier
    conditioning_model = ModelClass(tokenizer, n_classes, CLASSIFIER_DROPOUT)
    checkpoint = torch.load(ckpt, map_location=device)
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    return conditioning_model

# assert torch.cuda.device_count() >= 2, 'need at least two gpus'
# gpu0 = torch.device('cuda:0') # backbone
# gpu1 = torch.device('cuda:1') # two discriminator models
gpu0 = torch.device('cuda:0')
gpu1 = torch.device('cuda:0')

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(args.backbone, return_dict=True).to(gpu0)
    model.eval()
    no_length = args.ckpt_length is None

    conditioning_model_length = None if no_length else load_oracle_model_from_ckpt(args, gpu1, args.ckpt_length, tokenizer, args.length_num_classes)
    conditioning_model_label = load_oracle_model_from_ckpt(args, gpu1, args.ckpt_label, tokenizer, args.label_num_classes)
    
    combinations = [(False, True)] if no_length else [(True, True), (True, False), (False, True)]

    with open(args.output, 'w') as f:
        t1 = time.time()
        for _ in tqdm(range(args.num)):
            for allow_length, allow_label in combinations:

                if allow_length: length_level = args.length if args.length is not None else random.choice(range(args.length_num_classes))
                else: length_level = -1
                if allow_label: label = args.label if args.label is not None else random.choice(range(args.label_num_classes))
                else: label = -1
                length_level = torch.LongTensor([length_level])
                label = torch.LongTensor([label])
                input_prefix = BOS_TOKEN
                results = predict_oracle(model, 
                            tokenizer, 
                            conditioning_model_length,
                            conditioning_model_label, 
                            length_level,
                            label,
                            [input_prefix], 
                            precondition_topk=args.precondition_topk,
                            do_sample=args.do_sample,
                            length_cutoff=args.length_cutoff,
                            condition_lambda_length=args.condition_lambda_length,
                            condition_lambda_label=args.condition_lambda_label,
                            topk=args.topk,
                            no_condition_past=args.no_condition_past,
                            multiclass=args.multiclass,
                            device=args.device,
                            min_length=args.min_length)
                print(f'{label.item()}\t{length_level.item()}', file=f)
                postprocessed_str = results[0].replace('\n', '\\n')
                print(postprocessed_str, file=f)
        t2 = time.time()
        avg_time = (t2-t1) / args.num
        print(f'average time per sentence: {avg_time} seconds')

def predict_oracle(model, tokenizer, conditioning_model_length, conditioning_model_label, length_level, label, input_prefixes, precondition_topk=200, do_sample=False, length_cutoff=384, condition_lambda_length=1.0, condition_lambda_label=1.0, topk=50, no_condition_past=False, multiclass=False, device='cuda', min_length=0):
    with torch.no_grad():
        batch_size = len(input_prefixes)

        # assumes initially all same length. TODO: pad it (from the left?)
        input_ids = [tokenizer.encode(it, return_tensors='pt').to(device) for it in input_prefixes] # batch x seq
        input_ids = torch.cat(input_ids, dim=0)

        length_level = length_level.to(device)
        label = label.to(device)

        cur_len = input_ids.size(1)
        max_length = length_cutoff
        temperature = 1.0
        top_k = topk
        top_p = 1.0
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = None
        pad_token_id = tokenizer.encode(PAD_TOKEN)[0]
        eos_token_id = tokenizer.encode(EOT_TOKEN)[0]
        input_attention_mask = input_ids.new_ones(input_ids.shape)
        use_cache = True
        model_specific_kwargs = {}

        output = _generate_no_beam_search(model,
                                        conditioning_model_length,
                                        conditioning_model_label,
                                        condition_lambda_length,
                                        condition_lambda_label,
                                        length_level,
                                        label,
                                        precondition_topk,
                                        input_ids,
                                        cur_len,
                                        max_length,
                                        min_length,
                                        do_sample,
                                        temperature,
                                        top_k,
                                        top_p,
                                        repetition_penalty,
                                        no_repeat_ngram_size,
                                        bad_words_ids,
                                        pad_token_id,
                                        eos_token_id,
                                        batch_size,
                                        input_attention_mask,
                                        use_cache,
                                        no_condition_past,
                                        multiclass,
                                        model_specific_kwargs)

        return [tokenizer.decode(s) for s in output] # 1: to delete the pad token

def modify_condition_past(condition_past, top_indices, last_added_tokens):
    _, precondition_topk = top_indices.size(0), top_indices.size(1)
    for i in range(len(condition_past)): # 12
        for j in range(len(condition_past[0])): # 2
            to_modify = condition_past[i][j][:,:,-1,:] # (bz*pre_k,12,64)
            to_modify = to_modify.reshape(-1, precondition_topk, to_modify.size(1), to_modify.size(2)) # (bz,pre_k,12,64)
            to_keep = to_modify[(top_indices == last_added_tokens).nonzero(as_tuple=True)] # (bz,12,64)
            to_keep = to_keep.unsqueeze(1).expand(to_modify.size()) # (bz,pre_k,12,64)
            condition_past[i][j][:,:,-1,:] = to_keep.flatten(0, 1)
    
    return condition_past


# hack of code from transformers/generation_utils.py
# to get our conditioning
def _generate_no_beam_search(
        model,
        conditioning_model_length,
        conditioning_model_label,
        condition_lambda_length,
        condition_lambda_label,
        length_level,
        label,
        precondition_topk,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        input_attention_mask,
        use_cache,
        no_condition_past,
        multiclass,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1) # tensor([1,...,1]) size (bz,)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        condition_past_length = None
        condition_past_label = None
        last_top_indices = None
        while cur_len < max_length:
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=input_attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = model(**model_inputs, return_dict=True)
            scores = outputs.logits[:, -1, :] # logits size (bz, seqlen, vocab_size)

            if eos_token_id is not None and cur_len < min_length:
                scores[:, eos_token_id] = -float("inf")

            # scores = model.postprocess_next_token_scores(
            #     scores=next_token_logits,
            #     input_ids=input_ids,
            #     no_repeat_ngram_size=no_repeat_ngram_size,
            #     bad_words_ids=bad_words_ids,
            #     cur_len=cur_len,
            #     min_length=min_length,
            #     max_length=max_length,
            #     eos_token_id=eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     batch_size=batch_size,
            #     num_beams=1,
            # ) # scores size (bz, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            # !!!DEBUG
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            #------FUDGE discriminator related------#
            if ((condition_past_length is not None) or (condition_past_label is not None)) and (not no_condition_past):
                last_top_indices = top_indices
            top_logits, top_indices = scores.topk(precondition_topk, dim=1) # (bz,topk) and (bz,topk)
            lambda_condition_logits_length, condition_past_length = condition_model_step(input_ids, length_level, top_logits, top_indices, last_top_indices, condition_past_length, conditioning_model_length, condition_lambda_length, precondition_topk, no_condition_past, cur_len, batch_size, multiclass)
            lambda_condition_logits_label, condition_past_label = condition_model_step(input_ids, label, top_logits, top_indices, last_top_indices, condition_past_label, conditioning_model_label, condition_lambda_label, precondition_topk, no_condition_past, cur_len, batch_size, multiclass)

            full_logits = top_logits + lambda_condition_logits_length + lambda_condition_logits_label
            # if do_sample:
            #     raise NotImplementedError
            # else:
            #     # Greedy decoding
            #     next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(full_logits, dim=-1)]

            if do_sample:
                # # Temperature (higher temperature => more likely to sample low probability tokens)
                # if temperature != 1.0:
                #     full_logits = full_logits / temperature
                # # Top-p/top-k filtering
                # next_token_logscores = top_k_top_p_filtering(full_logits, top_k=top_k, top_p=top_p)
                # # Sample
                # probs = F.softmax(next_token_logscores, dim=-1)
                # next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                post_logits, post_indices = full_logits.topk(top_k, dim=1)
                post_probs = F.softmax(post_logits, dim=1)
                index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()]
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices]
            else:
                # Greedy decoding
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(full_logits, dim=-1)]

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            input_attention_mask = torch.cat(
                [input_attention_mask, input_attention_mask.new_ones((input_attention_mask.shape[0], 1))], dim=-1
            )

        return input_ids

def condition_model_step(input_ids, classification_labels, top_logits, top_indices, last_top_indices, condition_past, conditioning_model, condition_lambda, precondition_topk, no_condition_past, cur_len, batch_size, multiclass):
    if classification_labels == -1:
        return torch.zeros_like(top_logits), None
    input_ids = input_ids.to(gpu1)
    classification_labels = classification_labels.to(gpu1)
    top_logits = top_logits.to(gpu1)
    top_indices = top_indices.to(gpu1)
    last_top_indices = last_top_indices.to(gpu1) if last_top_indices is not None else last_top_indices
    
    # expanded_lengths = torch.LongTensor([[cur_len for _ in range(precondition_topk)] for _ in range(batch_size)]).to(scores.device)
    if condition_lambda == 0:
        condition_logits = torch.zeros_like(top_logits).float()
    else:
        if (condition_past is not None) and (not no_condition_past):
            condition_past = modify_condition_past(condition_past, last_top_indices, input_ids[:, -1])
            raw_condition_logits, condition_past = conditioning_model.inference(
                top_indices.unsqueeze(2).flatten(0, 1), # (bz*topk, 1)
                None,
                condition_past
            )
        else:
            # input_ids (bz,cur_len) -> (bz,topk,cur_len)
            # top_indices (bz,topk) -> (bz,topk,1)
            # [CAT] -> tplus1_candidates (bz,topk,cur_len+1)
            tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2)
            # # command_attention_mask (bz,cmd_len) [CAT] (bz,cur_len+1) -> (bz,cmd_len+cur_len+1) -> (bz,1,cmd_len+cur_len+1) -> (bz,topk,cmd_len+cur_len+1)
            # expanded_attention_mask = torch.cat([
            #     command_attention_mask, 
            #     command_attention_mask.new_ones((batch_size, cur_len+1))
            # ], dim=1).unsqueeze(1).expand(-1, precondition_topk, -1)

            raw_condition_logits, condition_past = conditioning_model.inference(
                tplus1_candidates.flatten(0, 1), # (bz*topk, cmd_len+cur_len+1)
                None, # TODO: no attention mask here means nothing is padded. This should be the case since initially there is just one [BOS] token
                None
            )
        if multiclass:
            condition_logits = torch.gather(
                raw_condition_logits,
                2,
                classification_labels.reshape(batch_size,1,1).expand(raw_condition_logits.size())
                )[:,-1,0] # (bz*topk,seqlen,n_classes) -> (bz*topk,seqlen=-1,n_classes=classification_labels)
        else:
            raw_condition_logits = raw_condition_logits.view(raw_condition_logits.size(0), raw_condition_logits.size(1), -1) # make sure raw_condition_logits is (bz*topk,n_classes,seqlen)
            condition_logits = torch.gather(
                raw_condition_logits,
                1,
                classification_labels.reshape(batch_size,1,1).expand(raw_condition_logits.size())
                )[:,0,-1] # (bz*topk,n_classes,seqlen) -> (bz*topk,n_classes=classification_labels,seqlen=-1)
        condition_logits = condition_logits.view(batch_size, precondition_topk) # (bz*topk) -> (bz,topk)
        condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs: log(sigmoid(x)) = x-log(1+exp(x))
    
    input_ids = input_ids.to(gpu0)
    classification_labels = classification_labels.to(gpu0)
    top_logits = top_logits.to(gpu0)
    top_indices = top_indices.to(gpu0)
    last_top_indices = last_top_indices.to(gpu0) if last_top_indices is not None else last_top_indices
    return condition_lambda * condition_logits.to(gpu0), condition_past

if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)