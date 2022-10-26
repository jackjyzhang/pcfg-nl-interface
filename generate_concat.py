import sys
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from argparse import ArgumentParser
import time
from tqdm import tqdm
import torch.nn.functional as F
import pickle

from constants import *
# from train_cls_or_baseline import load_model_no_state
from model import ConcatModel, FreezeConcatModel
from template2cmd import Template
from generate_alignment import sample_command, sample_label_length

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='concat', choices=['concat', 'concatoracle'])
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--min_length', type=int, default=10) # so that it won't produce empty string
    parser.add_argument('--show_special_tokens', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--topk', type=int, default=50, help='top k for sampling')

    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--label', type=int, default=None)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--output', type=str, default=None)

    parser.add_argument('--no_label', action='store_true', help='only consider commands without label')
    parser.add_argument('--no_length', action='store_true', help='only consider commands without length')
    parser.add_argument('--comprehensive', action='store_true', help='do all combinations of attributes * num. overrides --no_label and --no_length')
    parser.add_argument('--exclude_length', action='store_true', help='works with --comprehensive')

    # task
    parser.add_argument('--freeze', action='store_true', help='concatfreeze instead of concat')
    parser.add_argument('--oracle', action='store_true', help='concatoracle instead of concat')

    # dataset related
    parser.add_argument('--label_num_classes', type=int, default=5)
    parser.add_argument('--length_num_classes', type=int, default=5)

    # zero-shot
    parser.add_argument('--block_label_class', type=int, default=None)
    parser.add_argument('--block_length_class', type=int, default=None)

    # few-shot
    parser.add_argument('--noncomp_label_class', type=int, default=None)

    # DEBUG
    parser.add_argument('--debug_freeze_model', action='store_true', help='does not really freeze, to debug freeze model')
    parser.add_argument('--debug_freeze_pos_ids', action='store_true', help='if turned on, DOES NOT change position_ids, otherwise change position_ids')
    parser.add_argument('--show_attention', action='store_true')

    args = parser.parse_args()
    if args.oracle: args.task = 'concatoracle'
    if args.task == 'concatoracle': args.oracle = True

    return args

def load_concat_model(args, tokenizer):
    if not args.freeze:
        gpt2_model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_STRING, pad_token_id=tokenizer.encode(PAD_TOKEN)[0])
        gpt2_model.resize_token_embeddings(len(tokenizer)) # because we added [PAD] as special token
        return ConcatModel(gpt2_model)
    else:
        gpt2_model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_STRING, pad_token_id=tokenizer.encode(PAD_TOKEN)[0])
        gpt2_model.resize_token_embeddings(len(tokenizer)) # because we added [PAD] as special token
        change_position_ids = not args.debug_freeze_pos_ids
        return FreezeConcatModel(gpt2_model, change_position_ids)
# hack of code from transformers/generation_utils.py
# to get our conditioning
def _cmd_past_generate(
        model,
        input_ids,
        command_logits,
        command_past,
        command_len,
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
        use_cache,
        start_with_pad,
        save_attention,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        if save_attention:
            attn_list = []
        
        last_command_logits = command_logits[:, -1, :] # (bz, vocab_size)
        # # TODO: can be optimized
        # precondition_topk = last_command_logits.size(1) # TODO: use vocab_size? or smaller?
        # top_logits, top_indices = last_command_logits.topk(precondition_topk, dim=1)
        # post_logits, post_indices = top_logits.topk(top_k, dim=1)
        # post_probs = F.softmax(post_logits, dim=1)
        # index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()]
        # input_ids = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # (bz, 1)

        if start_with_pad:
            input_ids = pad_token_id * torch.ones((batch_size, 1), device=command_logits.device).long()
        if input_ids is None:
            # sample input id from last logits, if no `input_ids` is given and `start_with_pad` is false
            post_logits, post_indices = last_command_logits.topk(top_k, dim=1)
            post_probs = F.softmax(post_logits, dim=1)
            next_token_idx = torch.multinomial(post_probs, 1).flatten()
            input_ids = post_indices[torch.arange(batch_size).to(post_indices.device), next_token_idx].unsqueeze(-1)

        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1) # tensor([1,...,1]) size (bz,)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        input_attention_mask = torch.cat([
            torch.ones((batch_size, command_len), device=input_ids.device),
            torch.ones((batch_size, 1), device=input_ids.device) - int(start_with_pad)
        ], dim=1) # if start with pad, first position is masked

        cur_len = 1 # TODO: 1 or cmd_len + 1?
        past = command_past
        while cur_len < max_length:
            # __import__('pdb').set_trace()
            outputs = model(input_ids, attention_mask=input_attention_mask, past_key_values=past, use_cache=use_cache, output_attentions=save_attention)
            if save_attention:
                attn_list.append(outputs.attentions)
            scores = outputs.logits[:, -1, :] # logits size (bz, seqlen, vocab_size)

            if do_sample:
                post_logits, post_indices = scores.topk(top_k, dim=1)
                post_probs = F.softmax(post_logits, dim=1)
                next_token_idx = torch.multinomial(post_probs, 1).flatten()
                next_token = post_indices[torch.arange(batch_size).to(post_indices.device), next_token_idx]
            else:
                # Greedy decoding
                next_token = torch.argmax(scores, dim=-1)

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

        if save_attention:
            return input_ids, attn_list
        return input_ids, None

def do_generate(args, command_prefix, gpt2_tokenizer, tokenizer, gpt2_model, model):
    if args.freeze:
        # prefix = command_prefix[:-1] + ':'
        # inp = PAD_TOKEN
        # cmd_ids_with_semicolon = gpt2_tokenizer.encode(prefix, return_tensors='pt').to(args.device)
        # input_ids = tokenizer.encode(inp, return_tensors='pt').to(args.device)
        # attention_mask = torch.cat([torch.ones_like(cmd_ids), torch.zeros_like(input_ids)], dim=1) # (1,cmd_len+1)
        # attention_mask = torch.ones_like(input_ids)
        # cmd_ids = cmd_ids_with_semicolon[:, :-1]
        # input_ids = cmd_ids_with_semicolon[:, -1:]
        cmd_ids = gpt2_tokenizer.encode(command_prefix, return_tensors='pt').to(args.device)
        input_ids = None # will turn to pad
        out = gpt2_model(cmd_ids, output_attentions=args.show_attention)
        cmd_past = out.past_key_values
        cmd_logits = out.logits
        if args.show_attention:
            attn_cmd = out.attentions
        # topk_output = model.generate(
        #     input_ids, 
        #     attention_mask=attention_mask,
        #     do_sample=True, 
        #     max_length=args.max_length, 
        #     top_k=args.topk,
        #     # past_key_values=cmd_past
        # )
        # ignore_len = len(inp)
        ignore_len = 0 # ': '

        cmd_len = cmd_ids.size(1)
        max_length = args.max_length
        min_length = 0
        do_sample = True
        temperature = 1.0
        top_k = args.topk
        top_p = 1.0
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = None
        pad_token_id = tokenizer.encode(PAD_TOKEN)[0]
        eos_token_id = tokenizer.encode(EOT_TOKEN)[0]
        batch_size = cmd_ids.size(0) # always 1 actually
        use_cache = True
        start_with_pad = True
        save_attention = args.show_attention
        model_specific_kwargs = {}

        topk_output, attn_list = _cmd_past_generate(model,
                                    input_ids,
                                    cmd_logits,
                                    cmd_past,
                                    cmd_len,
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
                                    use_cache,
                                    start_with_pad,
                                    save_attention,
                                    model_specific_kwargs)
        if save_attention:
            tokens = tokenizer.convert_ids_to_tokens(topk_output[0].tolist())
            attn_tuple = tuple([attn_cmd] + attn_list)
            attention = tuple(tuple(attn.cpu() for attn in list(attn_t)) for attn_t in list(attn_tuple))
            with open('concatfreeze_attn.pkl', 'wb') as f:
                pickle.dump((tokens, attention), f, protocol=4)

    else:
        prefix = command_prefix[:-1] + ':' if args.task == 'concat' else command_prefix # change period to semicolon
        input_ids = tokenizer.encode(prefix, return_tensors='pt')
        input_ids = input_ids.to(args.device)
        generate_outputs = model.generate(
            input_ids, 
            do_sample=True, 
            max_length=args.max_length, 
            min_length=args.min_length + input_ids.size(1),
            top_k=args.topk,
            eos_token_id=tokenizer.encode(EOT_TOKEN)[0],
            return_dict_in_generate=True,
            output_attentions=args.show_attention
        )
        topk_output = generate_outputs.sequences
        if args.show_attention:
            tokens = tokenizer.convert_ids_to_tokens(topk_output[0].tolist())
            attention = tuple(tuple(attn.cpu() for attn in list(attn_t)) for attn_t in list(generate_outputs.attentions))
            with open('concat_attn.pkl', 'wb') as f:
                pickle.dump((tokens, attention), f)
        ignore_len = len(prefix) if args.task == 'concat' else 0
    return topk_output, ignore_len

def sample_concatoracle_command(label, length_level):
    label_command = f'[LABEL{label}]' if label != -1 else ''
    length_command = f'[LENGTH{length_level}]' if length_level != -1 else ''
    command_prefix = label_command+length_command
    return command_prefix

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    zero_shot_on = args.block_label_class is not None or args.block_length_class is not None

    if args.oracle:
        label_tokens = [f'[LABEL{i}]' for i in range(args.label_num_classes)]
        if not args.exclude_length: 
            length_tokens = [f'[LENGTH{i}]' for i in range(args.length_num_classes)]
        else: length_tokens = []
        tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_STRING, pad_token=PAD_TOKEN, additional_special_tokens=label_tokens+length_tokens)
    else:
        tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_STRING)
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN)[0]
    concat_model = load_concat_model(args, tokenizer)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        try:
            state_dict = {k[0:]:v for k,v in checkpoint['state_dict'].items()} # ignore fix
            concat_model.load_state_dict(state_dict)
        except:
            state_dict = {k[7:]:v for k,v in checkpoint['state_dict'].items()} # fix 'module.model.xxx' error (dump 'module.' prefix in state dict keys)
            concat_model.load_state_dict(state_dict)
    concat_model = concat_model.to(args.device)
    model = concat_model.model
    if args.freeze:
        if args.debug_freeze_model:
            # for debugging, use the same model (essentially does not freeze)
            gpt2_model = model
        else:
            gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2').to(args.device)
            gpt2_model.eval()
        gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        gpt2_model, gpt2_tokenizer = None, None # placeholder
    if args.output:
        f = open(args.output, 'w')
    else:
        f = sys.stdout

    if args.interactive:
        raise NotImplementedError
    
    else:
        template_obj = Template(args.template, tokenizer)
        t1 = time.time()
        if args.comprehensive:
            if args.exclude_length: combinations = [(False, True)]
            elif args.noncomp_label_class is not None:
                combinations = [(True, True)]
                args.label = args.noncomp_label_class
            else: 
                combinations = [(True, True), (True, False), (False, True)]

            if zero_shot_on:
                # half: in_zero_shot=True
                for _ in tqdm(range(args.num // 2)):
                    for allow_length, allow_label in combinations:
                        gen_label, gen_length = sample_label_length(args.block_label_class, args.block_length_class, True, args.label_num_classes, args.length_num_classes)
                        label, length_level, command_prefix = sample_command(template_obj, gen_label, gen_length, allow_length, allow_label, strict=True)
                        if args.oracle:
                            command_prefix = sample_concatoracle_command(label, length_level)
                        topk_output, ignore_len = do_generate(args, command_prefix, gpt2_tokenizer, tokenizer, gpt2_model, model)
                        
                        prefix = command_prefix if args.oracle else command_prefix[:-1] + ':'
                        print(f'{label}\t{length_level}\t{prefix}', file=f)
                        decoded_str = tokenizer.decode(topk_output[0], skip_special_tokens=not args.show_special_tokens)
                        postprocessed_str = decoded_str[ignore_len:].replace('\\n:', '\\n').replace('\n','\\n').strip() # does not show prefix in the front
                        print(postprocessed_str, file=f) 
                # half: in_zero_shot=False
                for _ in tqdm(range(args.num // 2)):
                    for allow_length, allow_label in combinations:
                        gen_label, gen_length = sample_label_length(args.block_label_class, args.block_length_class, False, args.label_num_classes, args.length_num_classes)
                        label, length_level, command_prefix = sample_command(template_obj, gen_label, gen_length, allow_length, allow_label, strict=True)
                        if args.oracle:
                            command_prefix = sample_concatoracle_command(label, length_level)
                        topk_output, ignore_len = do_generate(args, command_prefix, gpt2_tokenizer, tokenizer, gpt2_model, model)
                        
                        prefix = command_prefix if args.oracle else command_prefix[:-1] + ':'
                        print(f'{label}\t{length_level}\t{prefix}', file=f)
                        decoded_str = tokenizer.decode(topk_output[0], skip_special_tokens=not args.show_special_tokens)
                        postprocessed_str = decoded_str[ignore_len:].replace('\\n:', '\\n').replace('\n','\\n').strip() # does not show prefix in the front
                        print(postprocessed_str, file=f) 
            else:
                for _ in tqdm(range(args.num)):
                    for allow_length, allow_label in combinations:
                        label, length_level, command_prefix = sample_command(template_obj, args.label, args.length, allow_length, allow_label, strict=True)
                        if args.oracle:
                            command_prefix = sample_concatoracle_command(label, length_level)
                        topk_output, ignore_len = do_generate(args, command_prefix, gpt2_tokenizer, tokenizer, gpt2_model, model)
                        
                        prefix = command_prefix if args.oracle else command_prefix[:-1] + ':'
                        print(f'{label}\t{length_level}\t{prefix}', file=f)
                        decoded_str = tokenizer.decode(topk_output[0], skip_special_tokens=not args.show_special_tokens)
                        postprocessed_str = decoded_str[ignore_len:].replace('\\n:', '\\n').replace('\n','\\n').strip() # does not show prefix in the front
                        print(postprocessed_str, file=f) 
        else:
            for _ in tqdm(range(args.num)):
                label, length_level, command_prefix = sample_command(template_obj, args.label, args.length, not args.no_length, not args.no_label, strict=True)
                if args.oracle:
                    command_prefix = sample_concatoracle_command(label, length_level)
                topk_output, ignore_len = do_generate(args, command_prefix, gpt2_tokenizer, tokenizer, gpt2_model, model)
                    
                prefix = command_prefix if args.oracle else command_prefix[:-1] + ':'
                print(f'{label}\t{length_level}\t{prefix}', file=f)
                decoded_str = tokenizer.decode(topk_output[0], skip_special_tokens=not args.show_special_tokens)
                postprocessed_str = decoded_str[ignore_len:].replace('\\n', '\\n').replace('\n','\\n').strip() # does not show prefix in the front
                print(postprocessed_str, file=f)
        t2 = time.time()
        avg_time = (t2-t1) / args.num
        print(f'average time per sentence: {avg_time} seconds')
    f.close()






