import argparse
import sys

from sklearn.metrics import classification_report, confusion_matrix
sys.path.append('..')
from collections import Counter
import math
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk import word_tokenize, sent_tokenize
from constants import LANGUAGE_MODEL_STRING
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import torch
from template2cmd import Template
import os
from tqdm import tqdm
import numpy as np
import time
import mauve

device = torch.cuda.current_device() if torch.cuda.is_available() else -1
# PPL_TOKENIZER_STRING = 'EleutherAI/gpt-j-6B'
# PPL_MODEL_STRING = 'hivemind/gpt-j-6B-8bit'
PPL_TOKENIZER_STRING = 'EleutherAI/gpt-neo-2.7B'
PPL_MODEL_STRING = 'EleutherAI/gpt-neo-2.7B'

def path_wo_ext(path):
    return os.path.splitext(path)[0]

def text_entropy(sen_lis, k):
    #sen_lis is like [['i','am','you','</s>'] ...]
    #assume it is lowered case, and clean
    dd, num = {}, 0
    for sen in sen_lis:
        for i in range(0, len(sen) - k + 1):
            num += 1
            tt = ' '.join(sen[i:i+k])
            #print tt
            if not tt in dd: dd[tt] = 0
            dd[tt] += 1
    
    entro = 0.0
    for tt in dd:
        prob = float(dd[tt] * 1.0) / num
        entro = entro - math.log(prob) * prob
    return entro

def corpus_ref_bleu(refs, samples, num_gram):
    if num_gram == 2:
        weights = (0.5, 0.5)
    if num_gram == 3:
        weights = (0.333333, 0.333333, 1 - 0.333333 * 2)
    if num_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    """
    scores = []
    for kk, s in enumerate(samples):
        print(kk)
        if kk % int(len(samples) / 100) == 0 and kk > 0: 
            print(kk * 1, 'percent')
        score = nltk.translate.bleu_score.sentence_bleu(refs, s, weights, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return np.mean(scores)
    """
    refs_lis = [refs] * len(samples)
    score = nltk.translate.bleu_score.corpus_bleu(refs_lis, samples, weights, smoothing_function=SmoothingFunction().method1)
    return score

def corpus_self_bleu(samples, num_gram):
    if num_gram == 2:
        weights = (0.5, 0.5)
    if num_gram == 3:
        weights = (0.333333, 0.333333, 1 - 0.333333 * 2)
    if num_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    """
    scores = []
    for kk, s in enumerate(samples):
        print(kk)
        if kk % int(len(samples) / 100) == 0 and kk > 0: 
            print(kk * 1, 'percent')
        score = nltk.translate.bleu_score.sentence_bleu(refs, s, weights, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return np.mean(scores)
    """
    refs_lis = []
    for i in range(len(samples)):
        refs_lis.append(samples[:i] + samples[i+1:])
    score = nltk.translate.bleu_score.corpus_bleu(refs_lis, samples, weights, smoothing_function=SmoothingFunction().method1)
    return score

def break_text(text_list):
    sentences = []
    for text in text_list:
        sentences += sent_tokenize(text.lower())
    sentences = [word_tokenize(s) for s in sentences]
    return sentences

def load_generation_file(path):
    '''
    (index start from 0)
    even lines: get label and length_level, ignore raw command string
    odd lines: get generated text
    '''
    texts = []
    labels = []
    length_levels = []
    with open(path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if i % 2 == 0:
                line_seg = line.split('\t')
                if len(line_seg) == 3:
                    label, length_level, _ = line_seg
                else: # leng(line_seg) == 2
                    label, length_level = line_seg
                labels.append(int(label))
                length_levels.append(int(length_level))
            else: texts.append(line.replace('[PAD]', '').replace('[SEP]', '').replace('[BOS]', '').replace('<|endoftext|>', ''))
    
    print(f'total examples: {len(texts)}')
    if torch.all(torch.tensor(length_levels) == -1):
        no_length = True
    else:
        no_length = False
    
    filtered_texts = []
    filtered_labels = []
    filtered_length_levels = []
    for i in range(len(texts)):
        if len(texts[i]) > 0:
            filtered_texts.append(texts[i])
            filtered_labels.append(labels[i])
            filtered_length_levels.append(length_levels[i])
    print(f'non empty number of examples: {len(filtered_texts)}')
    return filtered_texts, filtered_labels, filtered_length_levels, no_length

def load_reference_file(path):
    texts = []
    with open(path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            texts.append(line.strip())
    return texts

def eval_diversity(texts, print_all=False):
    print('\n------evaluating diversity------')
    sents_list = break_text(texts)
    if print_all:
        for k in [1,2,3,4]:
            entropy = text_entropy(sents_list, k)
            print(f'{k}-gram entropy = {entropy:.4f}')
    else:
        k = 4
        entropy = text_entropy(sents_list, k)
        print(f'{k}-gram entropy = {entropy:.4f}')
    
    return entropy

def rep_ngram(sen_lis, num_gram=4):
    rep_lis = []
    for sen in sen_lis:
        uniq_ngram, all_ngram = {}, []
        for i in range(0, len(sen) - num_gram + 1):
            tt = ' '.join(sen[i:i + num_gram])
            if not tt in uniq_ngram: uniq_ngram[tt] = True
            all_ngram.append(tt)
        if len(all_ngram) == 0:
            print(f'warning: len(all_ngram) is 0!!! skipping... sample: {str(sen)}')
            continue
        rep = 1.0 - len(uniq_ngram) * 1.0 / len(all_ngram)
        rep_lis.append(rep)
    return np.mean(rep_lis)

def eval_repetition(texts, print_all=False):
    print('\n------evaluating repetition------')
    sents_list = break_text(texts)
    if print_all:
        for k in [1,2,3,4]:
            rep = rep_ngram(sents_list, k)
            print(f'{k}-gram rep = {rep:.4f}')
    else:
        k = 4
        rep = rep_ngram(sents_list, k)
        print(f'{k}-gram rep = {rep:.4f}')
    
    return rep

def eval_perplexity(model, tokenizer, texts, generation_file=None, block_str='', K=500, finetuned=False):
    if not torch.cuda.is_available():
        print('no GPU, so not evaluating perplexity')
        return -1
    nlls = []
    ppls = []
    lengths = []
    for i, text in enumerate(tqdm(texts, desc='perplexity')):
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        target_ids = input_ids.clone()
        try:
            outputs = model(input_ids, labels=target_ids)
        except:
            print('ppl model error')
            print(f'text=<{text}>')
            print(f'input_ids=<{input_ids}>')
            print(f'index: {i}')
            continue
        
        nll = outputs[0].cpu().detach()
        ppl = torch.exp(nll)
        length = input_ids.size(1)
        nlls.append(nll)
        ppls.append(ppl)
        lengths.append(length)
    nlls = torch.tensor(nlls).float()
    ppls = torch.tensor(ppls).float()
    lengths = torch.tensor(lengths).float()
    total_length = lengths.sum()

    token_avg_ppl = torch.exp(nlls.sum() / total_length)
    sent_avg_ppl = ppls.mean().item()
    weighted_avg_ppl = ((ppls * lengths).sum() / total_length).item()
    
    print(f'perplexity from model {PPL_MODEL_STRING} out of {len(ppls)}:')
    print(f'token_avg_ppl={token_avg_ppl:.4f}')
    print(f'sent_avg_ppl={sent_avg_ppl:.4f}')
    print(f'weighted_avg_ppl={weighted_avg_ppl:.4f}')

    if generation_file is not None:
        k = min(K, len(texts))
        # topk_ppls, topk_idx = (ppls * lengths).topk(k)
        topk_ppls, topk_idx = ppls.topk(k)
        if len(block_str) > 0:
            block_str = f'.{block_str}'
        ppl_suffix = '.pplft' if finetuned else '.ppl'
        with open(f'{path_wo_ext(generation_file)}{block_str}{ppl_suffix}', 'w') as f:
            print(f'token_avg_ppl={token_avg_ppl:.4f}', file=f)
            print(f'sent_avg_ppl={sent_avg_ppl:.4f}', file=f)
            print(f'weighted_avg_ppl={weighted_avg_ppl:.4f}', file=f)
            for i in range(len(topk_idx)):
                idx = topk_idx[i]
                print('-'*50, file=f)
                print(f'rank={i}', file=f)
                print(f'index={idx}', file=f)
                print(f'ppl={topk_ppls[i]}', file=f)
                print(f'length={lengths[idx]}', file=f)
                print(f'weight={lengths[idx]/total_length}', file=f)
                print(f'text={texts[idx]}', file=f)       

    return weighted_avg_ppl

def eval_mauve(ref, gen):
    if ref is None:
        return -1
    ref = load_reference_file(ref)
    gen = [text.replace('[BOS]', '').replace('<|endoftext|>', '') for text in gen] # get rid of BOS, EOS
    gen = [text for text in gen if len(text) > 0]
    print(f'number of mauve generations: {len(gen)}')
    if len(ref) < len(gen): print('WARNING: MAUVE #reference < #generated! They should be the same!')
    if len(ref) > len(gen):
        print('MAUVE #reference > #generated, truncating reference to have length #generated')
        ref = ref[:len(gen)]
    out = mauve.compute_mauve(p_text=ref, q_text=gen, device_id=0, max_text_length=512, verbose=False)
    print(f'MAUVE={out.mauve}')
    return out.mauve

def eval_quality(args, ppl_model, ppl_tok, texts, generation_file=None, block_str='', print_all=False):
    if args.reference is None:
        return -1, -1
    print('\n------evaluating quality------')
    avg_ppl = eval_perplexity(ppl_model, ppl_tok, texts, generation_file, block_str, finetuned=not args.orig_ppl)
    refs = load_reference_file(args.reference)
    if print_all:
        for n in [2,3,4]:
            bleu = corpus_ref_bleu(refs, texts, n)
            print(f'{n}-gram bleu = {bleu:.4f}')
    else:
        n = 4
        bleu = corpus_ref_bleu(refs, texts, n)
        print(f'{n}-gram bleu = {bleu:.4f}')
    mauve_score = eval_mauve(args.mauve_reference, texts)
    rep = eval_repetition(texts)
    return avg_ppl, bleu, mauve_score, rep

def get_acc(generated, ground_truth, tolerance=0):
    generated = torch.tensor(generated)
    ground_truth = torch.tensor(ground_truth)
    generated = generated[ground_truth != -1]
    ground_truth = ground_truth[ground_truth != -1]
    if ground_truth.size(0) == 0:
        return -1, -1 # no available label to calculate acc
    diff = (generated - ground_truth).abs()
    acc_at_tol = (diff <= tolerance).double().mean().item()
    mean_err = diff.double().mean().item()
    return acc_at_tol, mean_err

def get_per_class_acc(generated, ground_truth, masked_ground_truth=None, zs_class=-1, n_classes=-1):
    '''
    masked_ground_truth: only calculate acc for positions s.t. masked_ground_truth is -1
    (thus we are not computing compositional acc)
    '''
    assert n_classes != -1
    generated = torch.tensor(generated)
    ground_truth = torch.tensor(ground_truth)
    if masked_ground_truth is not None:
        masked_ground_truth = torch.tensor(masked_ground_truth)
        generated = generated[masked_ground_truth == -1]
        ground_truth = ground_truth[masked_ground_truth == -1]
    generated = generated[ground_truth != -1]
    ground_truth = ground_truth[ground_truth != -1]
    if ground_truth.size(0) == 0:
        return -1, -1, [-1]*n_classes # no available label to calculate acc
    print(confusion_matrix(ground_truth.cpu().numpy(), generated.cpu().numpy(), normalize='true'))
    print(classification_report(ground_truth.cpu().numpy(), generated.cpu().numpy()))
    acc = (generated == ground_truth).float().mean().item()
    if zs_class != -1:
        generated_nzs = generated[ground_truth != zs_class]
        ground_truth_nzs = ground_truth[ground_truth != zs_class]
        if ground_truth_nzs.size(0) == 0:
            nzs_acc = -1
        else:
            nzs_acc = (generated_nzs == ground_truth_nzs).float().mean().item()
    else:
        nzs_acc = acc
    acc_class = []
    for class_i in range(n_classes):
        generated_class = generated[ground_truth == class_i]
        ground_truth_class = ground_truth[ground_truth == class_i]
        if ground_truth_class.size(0) == 0:
            acc_class.append(-1)
        else:
            acc_i = (generated_class == ground_truth_class).float().mean().item()
            acc_class.append(acc_i)
    print(f'acc_class: {acc_class}')
    return acc, nzs_acc, acc_class

def print_bad_examples(texts, predicted, ground_truth, threshold=2):
    for i in range(len(texts)):
        if abs(predicted[i] - ground_truth[i]) >= threshold:
            print(f'gt={ground_truth[i]}, pred={predicted[i]}')
            print(texts[i])
            print()

def eval_sentiment(args, texts, ground_truth, mask_ground_truth=None, per_class=False, zs_class=-1):
    if args.sentiment is None:
        if per_class:
            return -1, [-1]*args.label_num_classes, None, None
        else:
            return (-1, -1, -1)
    print('\n------evaluating sentiment------')
    model = AutoModelForSequenceClassification.from_pretrained(args.sentiment)
    tokenizer = AutoTokenizer.from_pretrained(args.sentiment)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    LABEL_MAPPING_DICT = {
        'LABEL_0': 0,
        'LABEL_1': 1,
        'LABEL_2': 2,
        'LABEL_3': 3,
        'LABEL_4': 4,
    }
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    labels = [LABEL_MAPPING_DICT[result['label']] for result in pipe(texts, **tokenizer_kwargs)]
    if args.save_details:
        with open(args.sentiment_save_path, 'w') as f:
            print(','.join(labels), file=f)

    labels_count = Counter(labels)
    print(f'predicted sentiment labels = {labels_count.most_common()}')

    if per_class:
        acc, nzs_acc, acc_class = get_per_class_acc(labels, ground_truth, mask_ground_truth, zs_class, args.label_num_classes)
        return acc, nzs_acc, acc_class, labels, ground_truth
    else:
        acc, mean_err = get_acc(labels, ground_truth)
        acc_at_1, _ = get_acc(labels, ground_truth, tolerance=1)
        print(f'accuracy: {acc:.4f}')
        print(f'acc@tol=1: {acc_at_1:.4f}')
        print(f'mean err: {mean_err:.4f}')
        if args.print_bad_examples is not None: print_bad_examples(texts, labels, ground_truth, args.print_bad_examples)
        return acc, acc_at_1, mean_err

def eval_length(args, texts, ground_truth, mask_ground_truth=None, per_class=False, zs_class=-1, no_length=False):
    if no_length:
        if per_class:
            return -1, -1, [-1]*args.length_num_classes, None, None
        else:
            return -1, -1, -1
    print('\n------evaluating length------')
    tok = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_STRING)
    template_obj = Template(args.template, tok)
    def get_generated_length_level(length, tmpl=template_obj):
        for i, (target_len, wordslist) in enumerate(tmpl.len_transforms):
            # target_len is increasing. the last target_len is maxlen in dataset
            if length <= target_len:
                break
        # if length is larger than the last target_len, i = len(len_transforms) is still returned
        return i
    lengths = [len(tok(text)['input_ids']) for text in texts]
    if args.save_details:
        with open(args.lengths_save_path, 'w') as f:
            print(','.join(str(length) for length in lengths), file=f)

    print(f'average tokenized length = {sum(lengths) / len(lengths)}')
    length_level_hat = [get_generated_length_level(length) for length in lengths]

    if per_class:
        acc, nzs_acc, acc_class = get_per_class_acc(length_level_hat, ground_truth, mask_ground_truth, zs_class, args.length_num_classes)
        return acc, nzs_acc, acc_class, length_level_hat, ground_truth
    else:
        acc, mean_err = get_acc(length_level_hat, ground_truth)
        acc_at_1, _ = get_acc(length_level_hat, ground_truth, tolerance=1)
        print(f'accuracy: {acc:.4f}')
        print(f'acc@tol=1: {acc_at_1:.4f}')
        print(f'mean err: {mean_err:.4f}')
        if args.print_bad_examples is not None: print_bad_examples(texts, length_level_hat, ground_truth, args.print_bad_examples)
        return acc, acc_at_1, mean_err

def eval_composition(args, sent_generated, sent_ground_truth, len_generated, len_ground_truth, block_label_class=-1, block_length_class=-1, no_length=False):
    if args.sentiment is None or no_length:
        if block_label_class != -1 or block_length_class != -1:
            return -1, -1
        else:
            return -1
    sent_generated = torch.tensor(sent_generated)
    sent_ground_truth = torch.tensor(sent_ground_truth)
    len_generated = torch.tensor(len_generated)
    len_ground_truth = torch.tensor(len_ground_truth)
    
    valid_mask = torch.logical_and(sent_ground_truth != -1, len_ground_truth != -1)
    sent_generated = sent_generated[valid_mask]
    sent_ground_truth = sent_ground_truth[valid_mask]
    len_generated = len_generated[valid_mask]
    len_ground_truth = len_ground_truth[valid_mask]

    if block_label_class != -1 or block_length_class != -1:
        assert block_label_class == -1 or block_length_class == -1
        block_mask = torch.logical_or(sent_ground_truth == block_label_class, len_ground_truth == block_length_class) # if one of block is -1, does not matter because it has already been filtered by valid_mask
        unblock_mask = torch.logical_not(block_mask)

        sgb = sent_generated[block_mask]
        stb = sent_ground_truth[block_mask]
        lgb = len_generated[block_mask]
        ltb = len_ground_truth[block_mask]
        accb = torch.logical_and(sgb == stb, lgb == ltb).float().mean().item()

        sgu = sent_generated[unblock_mask]
        stu = sent_ground_truth[unblock_mask]
        lgu = len_generated[unblock_mask]
        ltu = len_ground_truth[unblock_mask]
        accu = torch.logical_and(sgu == stu, lgu == ltu).float().mean().item()

        print(f'comp-acc-bl: {accb:.4f}, comp-acc-unb: {accu:.4f}')
        return accb, accu

    else:
        correct = torch.logical_and(sent_generated == sent_ground_truth, len_generated == len_ground_truth)
        acc = correct.float().mean().item()
        print(f'comp-acc: {acc:.4f}')
        return acc

def split_index(labels, blocked_class):
    blocked_idx = []
    unblocked_idx = []
    for i in range(len(labels)):
        if labels[i] == blocked_class: blocked_idx.append(i)
        elif labels[i] != -1: unblocked_idx.append(i)
        # not using -1 labeled examples on zero-shot class to maintain balance
    return blocked_idx, unblocked_idx

def split_block(block_label_class, block_length_class, texts, labels, length_levels):
    '''split zero-shot and non-zero-shot examples'''
    assert block_label_class == -1 or block_length_class == -1, 'not doing zero-shot on label and length simultaneously'
    if block_label_class != -1:
        blocked_class = block_label_class
        candidate_labels = labels
    else:
        # NOTE: actually we can relax this assert. if not doing zero-shot blocked_idx will simply be empty, unblocked_idx contain all the idx
        assert block_length_class != -1, 'at least one of label and length need to be zero-shot'
        blocked_class = block_length_class
        candidate_labels = length_levels
    blocked_idx, unblocked_idx = split_index(candidate_labels, blocked_class)

    text_blocked = [texts[i] for i in blocked_idx]
    text_unblocked = [texts[i] for i in unblocked_idx]
    labels_blocked = [labels[i] for i in blocked_idx]
    labels_unblocked = [labels[i] for i in unblocked_idx]
    lengths_levels_blocked = [length_levels[i] for i in blocked_idx]
    lengths_levels_unblocked = [length_levels[i] for i in unblocked_idx]
    return text_blocked, text_unblocked, labels_blocked, labels_unblocked, lengths_levels_blocked, lengths_levels_unblocked

GPTNEO_AGNEWS='../backbone/gptneo_agnews_l256_boseos_ep1_lr0.00002'
GPTNEO_YELP='../backbone/gptneo_yelp_l200_boseos_ep1_lr0.00002'
def main_comprehensive(args):
    zero_shot_on = len(args.block_class) > 0
    line_results = []
    if args.reference is not None and device != -1:
        # only load ppl model once, a speed up
        if args.orig_ppl:
            ppl_model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_STRING).to(device)
            ppl_tok = AutoTokenizer.from_pretrained(PPL_TOKENIZER_STRING)
            ppl_model.eval()
        else:
            if args.dataset == 'agnews':
                ppl_model = AutoModelForCausalLM.from_pretrained(GPTNEO_AGNEWS).to(device)
                ppl_tok = AutoTokenizer.from_pretrained(GPTNEO_AGNEWS)
            else:
                ppl_model = AutoModelForCausalLM.from_pretrained(GPTNEO_YELP).to(device)
                ppl_tok = AutoTokenizer.from_pretrained(GPTNEO_YELP)
                ppl_model.eval()

    # format header
    acc_sent_header_str = ','.join([f'lab_acc_{i}' for i in range(args.label_num_classes)])
    acc_len_header_str = ','.join([f'len_acc_{i}' for i in range(args.length_num_classes)])
    if zero_shot_on:
        header = f'filename,ent_bl,ent_unb,ppl_bl,ppl_unb,bleu_bl,bleu_unb,rep_bl,rep_unb,mauve_bl,mauve_unb,lab_acc(nzs),{acc_sent_header_str},len_acc(nzs),{acc_len_header_str},acc_comp,acc_cb,acc_cu'
    else:
        header = f'filename,4-gram entropy,perplexity,bleu,rep,mauve,lab_acc,{acc_sent_header_str},len_acc,{acc_len_header_str},acc_comp'

    # main loop
    t1 = time.time()
    for i, generation_file in enumerate(args.generation_file):
        filename = os.path.basename(generation_file)
        print(f'---generation file: {filename}---')
        # split texts
        texts, labels, length_levels, no_length = load_generation_file(generation_file)
        if zero_shot_on:
            block_label_class, block_length_class = args.block_class[2*i], args.block_class[2*i+1]
            print(f'zero-shot on label={block_label_class}, length={block_length_class}')
            texts_blocked, texts_unblocked, labels_blocked, labels_unblocked, length_levels_blocked, length_levels_unblocked = split_block(block_label_class, block_length_class, texts, labels, length_levels)
        else: block_label_class, block_length_class = -1, -1
        
        # quality, diversity
        if zero_shot_on:
            ent_bl = eval_diversity(texts_blocked)
            ent_unb = eval_diversity(texts_unblocked)
            ppl_bl, bleu_bl, mauve_bl, rep_bl = eval_quality(args, ppl_model, ppl_tok, texts_blocked, generation_file, block_str='b')
            ppl_unb, bleu_unb, mauve_unb, rep_unb = eval_quality(args, ppl_model, ppl_tok, texts_unblocked, generation_file, block_str='u')
        else:
            entropy = eval_diversity(texts)
            avg_ppl, bleu, mauve, rep = eval_quality(args, ppl_model, ppl_tok, texts, generation_file)
        
        # sent
        acc_sent, acc_sent_nzs, acc_sent_class, sent_generated, sent_ground_truth = eval_sentiment(args, texts, labels, mask_ground_truth=length_levels, per_class=True, zs_class=block_label_class)
        
        # length
        acc_len, acc_len_nzs, acc_len_class, len_generated, len_ground_truth = eval_length(args, texts, length_levels, mask_ground_truth=labels, per_class=True, zs_class=block_length_class, no_length=no_length)

        # comp acc
        acc_comp = eval_composition(args, sent_generated, sent_ground_truth, len_generated, len_ground_truth, no_length=no_length)
        if zero_shot_on:
            acc_cb, acc_cu = eval_composition(args, sent_generated, sent_ground_truth, len_generated, len_ground_truth, block_label_class, block_length_class, no_length=no_length)
        
        acc_sent_class_str = ','.join([f'{acc_i:.4f}' for acc_i in acc_sent_class])
        acc_len_class_str = ','.join([f'{acc_i:.4f}' for acc_i in acc_len_class])

        # format result
        if zero_shot_on:
            line_str = f'{filename},{ent_bl:.4f},{ent_unb:.4f},{ppl_bl:.4f},{ppl_unb:.4f},{bleu_bl:.4f},{bleu_unb:.4f},{rep_bl:.4f},{rep_unb:.4f},{mauve_bl:.4f},{mauve_unb:.4f},{acc_sent_nzs:.4f},{acc_sent_class_str},{acc_len_nzs:.4f},{acc_len_class_str},{acc_comp:.4f},{acc_cb:.4f},{acc_cu:.4f}'
            zs_str='.zs'
        else:
            line_str = f'{filename},{entropy:.4f},{avg_ppl:.4f},{bleu:.4f},{rep:.4f},{mauve:.4f},{acc_sent:.4f},{acc_sent_class_str},{acc_len:.4f},{acc_len_class_str},{acc_comp:.4f}'
            zs_str=''
        with open(f'{path_wo_ext(generation_file)}{zs_str}.eval', 'w') as f:
            print(header, file=f)
            print(line_str, file=f)
        line_results.append(line_str)
    
    t2 = time.time()
    avg_time = (t2 - t1) / len(args.generation_file)
    print(f'number of files: {len(args.generation_file)}')
    print(f'average time per file: {avg_time} seconds')

    print(header)
    for line in line_results:
        print(line)

def main(args):
    line_results = []
    for generation_file in args.generation_file:
        if os.path.getsize(generation_file) == 0:
            continue
        filename = os.path.basename(generation_file)
        print(f'---generation file: {filename}---')
        texts, labels, length_levels, no_length = load_generation_file(generation_file)

        entropy = eval_diversity(texts)
        avg_ppl, bleu, mauve = eval_quality(args, texts) if args.reference is not None else (-1, -1)
        acc_s, acc_at_1_s, mean_err_s = eval_sentiment(args, texts, labels)
        acc_l, acc_at_1_l, mean_err_l = eval_length(args, texts, length_levels, no_length=no_length)
        line_str = f'{filename},{entropy:.4f},{avg_ppl:.4f},{bleu:.4f},{acc_s:.4f},{acc_at_1_s:.4f},{mean_err_s:.4f},{acc_l:.4f},{acc_at_1_l:.4f},{mean_err_l:.4f}'
        line_results.append(line_str)
    print('filename,4-gram entropy,perplexity,bleu,sent acc,sent acc@tol=1,sent mean err,len acc,len acc@tol=1,len mean err')
    for line in line_results:
        print(line)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, required=True, help='required by length')
    parser.add_argument('--generation_file', type=str, nargs='+', required=True)
    parser.add_argument('--reference', type=str, default=None, help='reference file to evaluate quality (bleu). leave empty to skip quality')
    parser.add_argument('--mauve_reference', type=str, default=None, help='reference for mauve, should be same length as generation')
    parser.add_argument('--sentiment', type=str, default=None, help='sentiment classifier path. leave empty to skip sentiment')

    parser.add_argument('--print_bad_examples', type=int, default=None)
    parser.add_argument('--save_details', action='store_true')
    parser.add_argument('--lengths_save_path', type=str, default='lengths.csv')
    parser.add_argument('--sentiment_save_path', type=str, default='sentiment.csv')

    # dataset related
    parser.add_argument('--dataset', type=str, choices=['yelp', 'agnews'], required=True)
    parser.add_argument('--label_num_classes', type=int, default=5)
    parser.add_argument('--length_num_classes', type=int, default=5)

    # for per-class
    parser.add_argument('--comprehensive', action='store_true')

    # orig_ppl: use not finetuned ppl
    parser.add_argument('--orig_ppl', action='store_true')

    # zero-shot
    # parser.add_argument('--block_label_class', type=int, nargs='*', default=[])
    # parser.add_argument('--block_length_class', type=int, nargs='*', default=[])
    parser.add_argument('--block_class', type=int, nargs='*', default=[])
    args = parser.parse_args()
    args.generation_file = [file for file in args.generation_file if os.path.getsize(file) > 0]
    args.generation_file = [file for file in args.generation_file if os.path.splitext(file)[1] == '.txt']
    print(f'files: {args.generation_file}')
    if len(args.block_class):
        assert len(args.block_class) == 2 * len(args.generation_file) # for each file , [1, -1] if sentiment block is 1 and length is not blocked
        
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.comprehensive:
        main_comprehensive(args)
    else:
        main(args)   
