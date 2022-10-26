from constants import BOS_TOKEN, EOT_TOKEN, SEPARATOR, CLASSIFIER_DROPOUT
from template2cmd import Template
import argparse
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import CommandAlignmentClassifier, FreezeCommandAlignmentClassifier
from predict_alignment import predict_alignment
from util import num_params
import time
from tqdm import tqdm

def create_fake_example(template_obj, label, length_level):
    assert label in template_obj.lab_transforms.keys(), f'label is {label}, template_obj.lab_transforms.keys() is {template_obj.lab_transforms.keys()}'
    assert length_level < len(template_obj.len_transforms)
    length_to_produce = template_obj.len_transforms[length_level][0] - 1
    return {
        'text': EOT_TOKEN*length_to_produce,
        'label': label,
        'entities': [],
        'keywords': []
    }

def sample_label_length(block_label_class, block_length_class, in_zero_shot, label_num_classes, length_num_classes):
    if in_zero_shot:
        return block_label_class, block_length_class
    else:
        if block_label_class is not None:
            candidates = list(range(label_num_classes))
            candidates.remove(block_label_class)
            label = random.choice(candidates)
        else: label = None
        if block_length_class is not None:
            candidates = list(range(length_num_classes))
            candidates.remove(block_length_class)
            length = random.choice(candidates)
        else: length = None
        return label, length

def sample_command(template_obj, gen_label, gen_length, allow_length, allow_label, strict=False, negate=False):
    label = gen_label if gen_label is not None else random.choice(list(template_obj.lab_transforms.keys()))
    length_level = gen_length if gen_length is not None else random.choice(range(len(template_obj.len_transforms)))
    fake_example = create_fake_example(template_obj, label, length_level)
    command_str, has_form, has_attr, has_label, has_length, content_form_str = template_obj.generate_single(
        fake_example, 
        allow_content=False, allow_length=allow_length, allow_label=allow_label, 
        return_metadata=True,
        negate=negate,
        strict=strict)
    label = label if has_label else -1
    length_level = length_level if has_length else -1
    return label, length_level, command_str


def main(args):
    zero_shot_on = args.block_label_class is not None or args.block_length_class is not None
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(args.backbone, return_dict=True).to(args.device)
    model.eval()

    ConditioningModelClass = FreezeCommandAlignmentClassifier if args.freeze else CommandAlignmentClassifier
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    conditioning_model = ConditioningModelClass(tokenizer, classifier_dropout=CLASSIFIER_DROPOUT)
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    template_obj = Template(args.template, tokenizer)
    with open(args.output, 'w') as f:
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
                        # command_prefix += SEPARATOR
                        input_prefix = BOS_TOKEN
                        results = predict_alignment(model, 
                                    tokenizer, 
                                    conditioning_model, 
                                    [command_prefix],
                                    [input_prefix], 
                                    precondition_topk=args.precondition_topk,
                                    do_sample=args.do_sample,
                                    length_cutoff=args.length_cutoff,
                                    condition_lambda=args.condition_lambda,
                                    topk=args.topk,
                                    no_condition_past=args.no_condition_past,
                                    device=args.device,
                                    freeze=args.freeze,
                                    min_length=args.min_length)
                        print(f'{label}\t{length_level}\t{command_prefix}', file=f)
                        postprocessed_str = results[0].replace('\n', '\\n')
                        print(postprocessed_str, file=f)
                # half: in_zero_shot=False
                for _ in tqdm(range(args.num // 2)):
                    for allow_length, allow_label in combinations:
                        gen_label, gen_length = sample_label_length(args.block_label_class, args.block_length_class, False, args.label_num_classes, args.length_num_classes)
                        label, length_level, command_prefix = sample_command(template_obj, gen_label, gen_length, allow_length, allow_label, strict=True)
                        # command_prefix += SEPARATOR
                        input_prefix = BOS_TOKEN
                        results = predict_alignment(model, 
                                    tokenizer, 
                                    conditioning_model, 
                                    [command_prefix],
                                    [input_prefix], 
                                    precondition_topk=args.precondition_topk,
                                    do_sample=args.do_sample,
                                    length_cutoff=args.length_cutoff,
                                    condition_lambda=args.condition_lambda,
                                    topk=args.topk,
                                    no_condition_past=args.no_condition_past,
                                    device=args.device,
                                    freeze=args.freeze,
                                    min_length=args.min_length)
                        print(f'{label}\t{length_level}\t{command_prefix}', file=f)
                        postprocessed_str = results[0].replace('\n', '\\n')
                        print(postprocessed_str, file=f)
            else:
                for _ in tqdm(range(args.num)):
                    for allow_length, allow_label in combinations:
                        label, length_level, command_prefix = sample_command(template_obj, args.label, args.length, allow_length, allow_label, strict=True)
                        # command_prefix += SEPARATOR
                        input_prefix = BOS_TOKEN
                        results = predict_alignment(model, 
                                    tokenizer, 
                                    conditioning_model, 
                                    [command_prefix],
                                    [input_prefix], 
                                    precondition_topk=args.precondition_topk,
                                    do_sample=args.do_sample,
                                    length_cutoff=args.length_cutoff,
                                    condition_lambda=args.condition_lambda,
                                    topk=args.topk,
                                    no_condition_past=args.no_condition_past,
                                    device=args.device,
                                    freeze=args.freeze,
                                    min_length=args.min_length)
                        print(f'{label}\t{length_level}\t{command_prefix}', file=f)
                        postprocessed_str = results[0].replace('\n', '\\n')
                        print(postprocessed_str, file=f)
        else:
            for _ in tqdm(range(args.num)):
                label, length_level, command_prefix = sample_command(template_obj, None, None, not args.no_length, not args.no_label, strict=True)
                # command_prefix += SEPARATOR
                input_prefix = BOS_TOKEN
                results = predict_alignment(model, 
                            tokenizer, 
                            conditioning_model, 
                            [command_prefix],
                            [input_prefix], 
                            precondition_topk=args.precondition_topk,
                            do_sample=args.do_sample,
                            length_cutoff=args.length_cutoff,
                            condition_lambda=args.condition_lambda,
                            topk=args.topk,
                            no_condition_past=args.no_condition_past,
                            device=args.device,
                            freeze=args.freeze,
                            min_length=args.min_length)
                print(f'{label}\t{length_level}\t{command_prefix}', file=f)
                postprocessed_str = results[0].replace('\n', '\\n')
                print(postprocessed_str, file=f)
        t2 = time.time()
        avg_time = (t2-t1) / args.num
        print(f'average time per sentence: {avg_time} seconds')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--label', type=int, default=None)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--output', type=str, required=True)

    parser.add_argument('--no_label', action='store_true', help='only consider commands without label')
    parser.add_argument('--no_length', action='store_true', help='only consider commands without length')
    parser.add_argument('--comprehensive', action='store_true', help='do all combinations of attributes * num. overrides --no_label and --no_length')
    parser.add_argument('--exclude_length', action='store_true', help='works with --comprehensive')


    # task
    parser.add_argument('--freeze', action='store_true', help='alignfreeze instead of alignment')

    # models
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)

    # decode config
    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')
    parser.add_argument('--min_length', type=int, default=0, help='min length')
    parser.add_argument('--topk', type=int, default=50, help='top k for sampling')

    # misc
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--no_condition_past', action='store_true')

    # dataset related
    parser.add_argument('--label_num_classes', type=int, default=5)
    parser.add_argument('--length_num_classes', type=int, default=5)

    # zero-shot
    parser.add_argument('--block_label_class', type=int, default=None)
    parser.add_argument('--block_length_class', type=int, default=None)

    # few-shot
    parser.add_argument('--noncomp_label_class', type=int, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)