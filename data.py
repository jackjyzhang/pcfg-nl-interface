'''adapted from https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation'''
import random
import math
import os
import pickle
from collections import defaultdict, namedtuple
import string
import logging

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # turn off since we're using multiple threads for loading anyway

from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import torch

from constants import *
from template2cmd import Template

from util import pad_to_maxlen

def collate_oracle(batch):
    # example = (pad_id, text_pt, length_label, classification_label)
    pad_id = batch[0][0]
    texts = [b[1] for b in batch]

    input_ids, attention_mask = pad_to_maxlen(texts, pad_id)

    length_labels = [b[2] for b in batch]
    assert type(length_labels[0]) == int
    length_labels = torch.LongTensor(length_labels)

    classification_labels = [b[3] for b in batch]
    assert type(classification_labels[0]) == int
    classification_labels = torch.LongTensor(classification_labels)

    return (input_ids, attention_mask, length_labels, classification_labels)

def collate_alignfreeze(batch):
    # example = (pad_id, separator_id, text_pt, command_pt, false_command_pt, classification_label, self.parent.use_machine_text, machine_text_pt)
    pad_id = batch[0][0]
    sep_id = batch[0][1]
    texts = [b[2] for b in batch]
    true_commands = [b[3] for b in batch]
    false_commands = [b[4] for b in batch]
    use_machine_text = batch[0][6]
    machine_texts = [b[7] for b in batch]
    labels, texts_commands_tuple = sample_labels(use_machine_text, texts, machine_texts, true_commands, false_commands)
    cmd_sep = []
    txts = []
    prefix_lengths = []
    text_lengths = []

    for i in range(len(labels)):
        txt, cmd = texts_commands_tuple[i]
        cmd_sep.append(torch.cat([cmd], dim=0))
        txts.append(txt)
        prefix_lengths.append(cmd.size(0)) # no +1 is for no separator
        text_lengths.append(txt.size(0))

    cmd_ids, cmd_attention_mask = pad_to_maxlen(cmd_sep, EOT_ID)
    text_ids, text_attention_mask = pad_to_maxlen(txts, pad_id)
    prefix_lengths = torch.LongTensor(prefix_lengths)
    text_lengths = torch.LongTensor(text_lengths)

    classification_labels = [b[5] for b in batch]
    assert type(classification_labels[0]) == int
    classification_labels = torch.LongTensor(classification_labels)

    return (cmd_ids, cmd_attention_mask, text_ids, text_attention_mask, labels, prefix_lengths, text_lengths, classification_labels)

def collate_alignment(batch):
    # example = (pad_id, separator_id, text_pt, command_pt, false_command_pt, classification_label, self.parent.use_machine_text, machine_text_pt)
    pad_id = batch[0][0]
    sep_id = batch[0][1]
    texts = [b[2] for b in batch]
    true_commands = [b[3] for b in batch]
    false_commands = [b[4] for b in batch]
    use_machine_text = batch[0][6]
    machine_texts = [b[7] for b in batch]
    labels, texts_commands_tuple = sample_labels(use_machine_text, texts, machine_texts, true_commands, false_commands)
    cmd_sep_txt = []
    prefix_lengths = []
    text_lengths = []

    for i in range(len(labels)):
        txt, cmd = texts_commands_tuple[i]
        cmd_sep_txt.append(torch.cat([cmd, (sep_id*torch.ones(1)).long(), txt], dim=0))
        prefix_lengths.append(cmd.size(0) + 1) # +1 is for separator
        text_lengths.append(txt.size(0))

    input_ids, attention_mask = pad_to_maxlen(cmd_sep_txt, pad_id)
    prefix_lengths = torch.LongTensor(prefix_lengths)
    text_lengths = torch.LongTensor(text_lengths)

    classification_labels = [b[5] for b in batch]
    assert type(classification_labels[0]) == int
    classification_labels = torch.LongTensor(classification_labels)

    return (input_ids, attention_mask, labels, prefix_lengths, text_lengths, classification_labels)

def sample_labels(use_machine_text, texts, machine_texts, true_commands, false_commands, machine_text_rate=0.5):
    '''machine_text_rate: probability of using machine-generated text'''
    bz = len(texts)
    labels = torch.randint(0,2,(bz,)) # sample label p(1) = p(0) = 0.5
    selected_texts_commands = []
    for i in range(len(labels)):
        label = labels[i]
        if label == 1:
            selected_texts_commands.append((texts[i], true_commands[i]))
        else:
            # negative example
            use_machine_text_flag = use_machine_text and (random.random() < machine_text_rate)
            # case 1: machine text, true command
            if use_machine_text_flag:
                selected_texts_commands.append((machine_texts[i], true_commands[i]))
            # case 2: real text, false command
            else:
                selected_texts_commands.append((texts[i], false_commands[i]))
    
    return labels, selected_texts_commands


def collate_concatfreeze(batch):
    pad_id = batch[0][0]
    cmd = [b[1] for b in batch]
    txt = [b[2] for b in batch]

    cmd_ids, cmd_attention_mask = pad_to_maxlen(cmd, EOT_ID)
    text_ids, text_attention_mask = pad_to_maxlen(txt, pad_id)
    return cmd_ids, cmd_attention_mask, text_ids, text_attention_mask

def collate_concat(batch):
    # example = (pad_id, cmd_txt_pt, text_start_pos, classification_label)
    pad_id = batch[0][0]
    cmd_txt = [b[1] for b in batch]
    text_start_pos = [b[2] for b in batch]

    lengths = [ct.size(0) for ct in cmd_txt]
    attn_masks = []
    loss_masks = [] # not shifted here. shifted left (along with labels=input_ids) when calculating loss

    max_length = max(lengths)
    for i in range(len(lengths)):
        if lengths[i] < max_length:
            cmd_txt[i] = torch.cat([cmd_txt[i], pad_id * torch.ones(max_length - lengths[i]).long()], dim=0)
            attn_masks.append(torch.cat([torch.ones(lengths[i]).long(), torch.zeros(max_length - lengths[i]).long()], dim=0))
            loss_masks.append(
                torch.cat([
                    torch.zeros(text_start_pos[i]).long(),
                    torch.ones(lengths[i] - text_start_pos[i]).long(),
                    torch.zeros(max_length - lengths[i]).long()
                ], dim=0)
            )
        else:
            attn_masks.append(torch.ones(max_length).long())
            loss_masks.append(
                torch.cat([
                    torch.zeros(text_start_pos[i]).long(), 
                    torch.ones(lengths[i] - text_start_pos[i]).long()
                ], dim=0)
            )
    
    input_ids = torch.stack(cmd_txt, dim=0)
    attention_masks = torch.stack(attn_masks, dim=0)
    loss_masks = torch.stack(loss_masks, dim=0)

    classification_labels = [b[3] for b in batch]
    assert type(classification_labels[0]) == int
    classification_labels = torch.LongTensor(classification_labels)

    return (input_ids, attention_masks, loss_masks, classification_labels)

def load_machine_gen(path):
    if path is None:
        return None
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
            else: texts.append(line.replace('[BOS]', '').replace('<|endoftext|>', '')) # get rid of potential special symbols

    text_dict = defaultdict(list)
    for i in range(len(texts)):
        text, label, length_level = texts[i], labels[i], length_levels[i]
        text_dict[(label, length_level)].append(text)

    return dict(text_dict)

class Dataset:
    def __init__(self, args, separator=SEPARATOR):
        print('loading data')
        random.seed(args.seed)
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.alignment = args.task == 'alignment'
        self.concat = args.task == 'concat'
        self.oracle = args.task.startswith('oracle') # 'oracle_length or oracle_label
        self.alignfreeze = args.task == 'alignfreeze'
        self.concatfreeze = args.task == 'concatfreeze'
        self.concatoracle = args.task == 'concatoracle'
        self.use_content_form = args.use_content_form # use automatically extracted keywords + NER
        self.use_length = args.use_length
        self.gpt2tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_STRING)
        self.use_machine_text = args.machine_gen is not None
        self.machine_text_dict = load_machine_gen(args.machine_gen)
        self.max_length = args.max_length
        self.label_num_classes = args.label_num_classes
        self.length_num_classes = args.length_num_classes

        # zero-shot
        self.block_label_class = args.block_label_class
        self.block_length_class = args.block_length_class
        self.block_completely = args.block_completely

        # compositional
        self.noncomp_label_class = args.noncomp_label_class

        if self.concat or self.concatfreeze:
            self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_STRING)
            self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        elif self.concatoracle:
            label_tokens = [f'[LABEL{i}]' for i in range(self.label_num_classes)]
            length_tokens = [f'[LENGTH{i}]' for i in range(self.length_num_classes)] if self.use_length else []
            self.tokenizer =AutoTokenizer.from_pretrained(LANGUAGE_MODEL_STRING, pad_token=PAD_TOKEN, additional_special_tokens=label_tokens+length_tokens)
        elif self.alignment or self.oracle or self.alignfreeze:
            self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_STRING, bos_token=BOS_TOKEN)
            self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
            self.tokenizer.add_special_tokens({'sep_token': separator})
            self.separator_id = self.tokenizer.encode(self.tokenizer.sep_token)[0]
        else:
            raise NotImplementedError
        self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.template_obj = Template(args.template_file, tokenizer=self.tokenizer)
        self.splits = {} # each value is a huggingface dataset

        # yelp review dataset only has train (650000) and test (50000)
        # a val size of (50000) is manually sampled from train. so the
        # final split is (600000, 50000, 50000)
        dataset = load_from_disk(self.data_dir)
        self.splits['train'], self.splits['val'], self.splits['test'] = dataset['train'], dataset['val'], dataset['test']

        # related to extra data
        self.extra_data = []
        self.data_weights = [len(self.splits['train'])] # because the original dataset length is in front, data_weights is offset by 1
        max_weight = self.data_weights[0] # examples-proportional mixing (Raffel et al., 2020), set artificial max dataset length to be the length of original dataset
        if len(args.extra_data) > 0:
            for i in range(0, len(args.extra_data), 2):
                data_path = args.extra_data[i]
                template_path = args.extra_data[i+1]
                dataset = load_from_disk(data_path)
                train_dataset = dataset['train']
                template_obj = Template(template_path, tokenizer=self.tokenizer)
                self.extra_data.append((train_dataset, template_obj))
                self.data_weights.append(min(len(train_dataset), max_weight))

        print('done loading data')
        print('split sizes:')
        for key in ['train', 'val', 'test']:
            print(key, len(self.splits[key]))
        if len(self.extra_data) > 0:
            print('extra data:')
            print('size\tpath\ttemplate')
            for i in range(len(self.extra_data)):
                print(f'{len(self.extra_data[i][0])}\t{args.extra_data[i*2]}\t{args.extra_data[i*2+1]}')

    def shuffle(self, split, seed=None):
        assert split in ['train', 'val', 'test']
        self.splits[split].shuffle(seed)


    def loader(self, split, num_workers=20, indices=None):
        assert split in ['train', 'val', 'test']
        data = self.splits[split] if indices is None else self.splits[split].select(indices)
        if self.concat or self.concatoracle:
            collate_choice = collate_concat
        elif self.concatfreeze:
            collate_choice = collate_concatfreeze
        elif self.oracle:
            collate_choice = collate_oracle
        elif self.alignfreeze:
            collate_choice = collate_alignfreeze
        else:
            collate_choice = collate_alignment
        return torch.utils.data.DataLoader(SplitLoader(data, split=='train', self), batch_size=self.batch_size, pin_memory=True, collate_fn=collate_choice, num_workers=num_workers)


class SplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, data, is_train, parent):
        super(SplitLoader).__init__()
        self.data = data
        self.pos = 0
        self.parent = parent
        self.is_train = is_train


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        return self
    

    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self):
                raise StopIteration

            # original example
            raw_example = self.data[self.pos] # huggingface dataset entry, keys = ['entities', 'keywords', 'label', 'text']
            template_obj = self.parent.template_obj

            # get length label    
            raw_length = len(self.parent.tokenizer.encode(raw_example['text'], return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0])
            for length_label, (len_cutoff, _) in enumerate(template_obj.len_transforms):
                    if raw_length < len_cutoff:
                        break
            classification_label = raw_example['label']

            # check for zero-shot
            label_blocked = classification_label in self.parent.block_label_class
            length_blocked = length_label in self.parent.block_length_class
            use_length = self.parent.use_length and not length_blocked
            use_label = not label_blocked
            if ((not use_length) and (not use_label)) or ((label_blocked or length_blocked) and self.parent.block_completely):
                # discard this training example completely if:
                # 1. block_completely flag is true
                # 2. neither length nor label is allowed
                valid = False
                self.pos += increment
                continue

            # check for compositional
            if classification_label in self.parent.noncomp_label_class:
                # either not use label or not use length
                if random.random() < 0.5:
                    use_length = False
                else:
                    use_label = False

            # check if use extra data instead. this only happens in training
            if self.is_train:
                dataset_idx = random.choices(range(len(self.parent.data_weights)), weights=self.parent.data_weights)[0]
                if dataset_idx != 0:
                    # use extra data
                    dataset_to_use, template_obj = self.parent.extra_data[dataset_idx-1] # i-1 is offset the weight on original dataset
                    raw_example = random.choice(dataset_to_use)

            if self.parent.alignment or self.parent.alignfreeze:
                raw_text = BOS_TOKEN + raw_example['text'] + EOT_TOKEN
                text_pt = self.parent.tokenizer.encode(raw_text, return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0] # might exceed model max length
                if self.parent.use_machine_text:
                    machine_text = BOS_TOKEN + random.choice(self.parent.machine_text_dict[(classification_label, length_label)]) + EOT_TOKEN
                    machine_text_pt = self.parent.tokenizer.encode(machine_text, return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0]
                else:
                    machine_text_pt = None

                
                command_tokenizer = self.parent.tokenizer if self.parent.alignment else self.parent.gpt2tokenizer
                command = template_obj.generate_single(
                    raw_example, allow_content=self.parent.use_content_form, allow_length=use_length, allow_label=use_label)
                command_pt = command_tokenizer.encode(command, return_tensors='pt')[0]
                false_command = template_obj.generate_single(
                    raw_example, allow_content=self.parent.use_content_form, allow_length=use_length, allow_label=use_label, negate=True)
                false_command_pt = command_tokenizer.encode(false_command, return_tensors='pt')[0]
                example = (self.parent.pad_id, self.parent.separator_id, text_pt, command_pt, false_command_pt, classification_label, self.parent.use_machine_text, machine_text_pt)
                valid = True
            
            elif self.parent.oracle:
                raw_text = BOS_TOKEN + raw_example['text'] + EOT_TOKEN
                text_pt = self.parent.tokenizer.encode(raw_text, return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0] # might exceed model max length
                example = (self.parent.pad_id, text_pt, length_label, classification_label)
                valid = True

            elif self.parent.concatoracle:
                _, _, _, has_label, has_length, _ = template_obj.generate_single(
                    raw_example, allow_content=self.parent.use_content_form, allow_length=use_length, allow_label=use_label, return_metadata=True)
                raw_text = raw_example['text']
                label_command = f'[LABEL{classification_label}]' if use_label and has_label else ''
                length_command = f'[LENGTH{length_label}]' if use_length and has_length else ''
                command = label_command+length_command # non-natural language command, mapped to a special token [LABEL1], [LABEL2], ...
                cmd_txt = f'{command} {raw_text}' + EOT_TOKEN
                cmd_txt_pt = self.parent.tokenizer.encode(cmd_txt, return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0] # might exceed model max length
                text_start_pos = 1
                example = (self.parent.pad_id, cmd_txt_pt, text_start_pos, classification_label)
                valid = True

            elif self.parent.concat or self.parent.concatfreeze:
                raw_text = raw_example['text']
                command = template_obj.generate_single(
                    raw_example, allow_content=self.parent.use_content_form, allow_length=use_length, allow_label=use_label)
                text_start_pos = len(self.parent.tokenizer.encode(command)) # encode command only to get the encoded length of command => start pos for text (there is not EOS)

                if self.parent.concatfreeze:
                    txt = PAD_TOKEN + raw_text + EOT_TOKEN
                    cmd_pt = self.parent.gpt2tokenizer.encode(command, return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0]
                    txt_pt = self.parent.tokenizer.encode(txt, return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0]
                    example = (self.parent.pad_id, cmd_pt, txt_pt)
                else:
                    command = command[:-1] + ':' # e.g. give me a review: xxx
                    cmd_txt = f'{command} {raw_text}' + EOT_TOKEN # need this so that generation can stop!! it's not added by default
                    cmd_txt_pt = self.parent.tokenizer.encode(cmd_txt, return_tensors='pt', truncation=True, max_length=self.parent.max_length)[0] # might exceed model max length
                    example = (self.parent.pad_id, cmd_txt_pt, text_start_pos, classification_label)

                valid = True

            else:
                raise NotImplementedError
            
            # temp debug
            # print(classification_label, length_label)
            # print(use_label, use_length)
            # print(command)
            # end temp debug
            self.pos += increment
        return example