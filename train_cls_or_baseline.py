'''adapted from https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation'''
import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel

from data import Dataset
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from constants import *
from model import ConcatModel, CommandAlignmentClassifier, OracleBinaryClassifier, OracleClassifier, FreezeCommandAlignmentClassifier, FreezeConcatModel

import wandb

def train(model, dataset, optimizer, scheduler, epoch, args, data_start_index, gpt2_model=None):
    model.train()
    if data_start_index == 0:
        dataset.shuffle('train', seed=epoch + args.seed)
    if args.epoch_max_len is not None:
        data_end_index = min(data_start_index + args.epoch_max_len, len(dataset.splits['train']))
        loader = dataset.loader('train', num_workers=args.num_workers, indices=list(range(data_start_index, data_end_index)))
        data_start_index = data_end_index if data_end_index < len(dataset.splits['train']) else 0
    else:
        loader = dataset.loader('train', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    acc_meter = AverageMeter('acc', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter, acc_meter], prefix='Training: ')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        if len(batch[0]) != args.batch_size:
            continue
        batch = [tensor.to(args.device) for tensor in batch]
        if args.task == 'concat' or args.task == 'concatoracle':
            input_ids, attention_masks, loss_masks, classification_labels = batch
            loss, n_loss = model(input_ids, attention_masks, loss_masks)
            loss = (loss * n_loss).sum() / n_loss.sum()
            n_loss = n_loss.sum()
            acc = torch.tensor(-1) # dummy value
        elif args.task == 'concatfreeze':
            cmd_ids, cmd_attention_mask, text_ids, text_attention_mask = batch
            out = gpt2_model(cmd_ids, attention_mask=cmd_attention_mask)
            cmd_past = out.past_key_values
            loss, n_loss = model(cmd_past, text_ids, cmd_attention_mask, text_attention_mask)
            acc = torch.tensor(-1) # dummy value
        elif args.task == 'alignment':
            input_ids, attention_mask, labels, prefix_lengths, text_lengths, classification_labels = batch
            loss, acc, n_loss = model(input_ids, attention_mask, prefix_lengths, text_lengths, labels)
            if not args.no_dataparallel:
                loss = (loss * n_loss).sum() / n_loss.sum()
                acc = (acc * n_loss).sum() / n_loss.sum()
                n_loss = n_loss.sum()
        elif args.task == 'alignfreeze':
            cmd_ids, cmd_attention_mask, text_ids, text_attention_mask, labels, prefix_lengths, text_lengths, classification_labels = batch
            out = gpt2_model(cmd_ids, attention_mask=cmd_attention_mask)
            cmd_past = out.past_key_values
            loss, acc, n_loss = model(cmd_past, text_ids, cmd_attention_mask, text_attention_mask, prefix_lengths, text_lengths, labels)
        elif args.task.startswith('oracle'):
            input_ids, attention_mask, length_labels, classification_labels = batch
            labels = length_labels if args.task == 'oracle_length' else classification_labels # oracle_label
            loss, acc, n_loss = model(input_ids, attention_mask, labels)
        else:
            raise NotImplementedError
   
        optimizer.zero_grad()
        loss.backward()
        if args.clip is not None: nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): scheduler.step()
        loss_meter.update(loss.detach(), int(n_loss))
        acc_meter.update(acc.detach(), int(n_loss))
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
            wandb.log({
                'batch_train_loss': loss_meter.val, 
                'batch_train_loss_avg': loss_meter.avg,
                'batch_train_acc_avg': acc_meter.avg
            })
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): scheduler.step()
    progress.display(total_length)
    wandb.log({'epoch_train_loss': loss_meter.avg, 'epoch_train_acc': acc_meter.avg, 'epoch': epoch})
    return data_start_index


def validate(model, dataset, criterion, epoch, args, per_class_acc=False, gpt2_model=None):
    model.eval()
    random.seed(0)
    loader = dataset.loader('val', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    acc_meter = AverageMeter('acc', ':6.4f')
    total_length = len(loader)
    if per_class_acc:
        acc0_meter = AverageMeter('acc0', ':6.4f')
        acc1_meter = AverageMeter('acc1', ':6.4f')
        acc_pos_meters = [AverageMeter(f'accp{i}', ':6.4f') for i in range(0,350,5)] # 0,5,...,345
        progress = ProgressMeter(total_length, [loss_meter, acc_meter, acc0_meter, acc1_meter]+acc_pos_meters, prefix='Validation: ')
    else:
        progress = ProgressMeter(total_length, [loss_meter, acc_meter], prefix='Validation: ')
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
            if len(batch[0]) != args.batch_size:
                continue
            batch = [tensor.to(args.device) for tensor in batch]
            if args.task == 'concat' or args.task == 'concatoracle':
                input_ids, attention_masks, loss_masks, classification_labels = batch
                loss, n_loss = model(input_ids, attention_masks, loss_masks)
                loss = (loss * n_loss).sum() / n_loss.sum()
                n_loss = n_loss.sum()
                acc = torch.tensor(-1) # dummy value
            elif args.task == 'concatfreeze':
                cmd_ids, cmd_attention_mask, text_ids, text_attention_mask = batch
                out = gpt2_model(cmd_ids, attention_mask=cmd_attention_mask)
                cmd_past = out.past_key_values
                loss, n_loss = model(cmd_past, text_ids, cmd_attention_mask, text_attention_mask)
                acc = torch.tensor(-1) # dummy value
            elif args.task == 'alignment':
                input_ids, attention_mask, labels, prefix_lengths, text_lengths, classification_labels = batch
                if per_class_acc:
                    assert args.no_dataparallel # currently per_class_acc does not support multi-card
                    loss, acc, n_loss, acc0, acc1, n0, n1, predicted_labels, flat_expanded_labels = model(input_ids, attention_mask, prefix_lengths, text_lengths, labels, per_class_acc=True)
                    predicted_labels = predicted_labels.reshape(args.batch_size, -1)
                    expanded_labels = flat_expanded_labels.reshape(args.batch_size, -1)
                    per_pos_acc = (predicted_labels == expanded_labels).float().mean(axis=0).cpu().tolist()
                    # usable_pos = loss_mask.all(axis=0).cpu().to_list() just also calculate acc for pad and prefix. it's easier
                    for i in range(0,min(350,expanded_labels.size(1)),5):
                        acc_pos_meters[i//5].update(per_pos_acc[i], args.batch_size)
                    acc0_meter.update(acc0.detach(), int(n0))
                    acc1_meter.update(acc1.detach(), int(n1))
                else:
                    loss, acc, n_loss = model(input_ids, attention_mask, prefix_lengths, text_lengths, labels)
                    if not args.no_dataparallel:
                        loss = (loss * n_loss).sum() / n_loss.sum()
                        acc = (acc * n_loss).sum() / n_loss.sum()
                        n_loss = n_loss.sum()
            elif args.task == 'alignfreeze':
                cmd_ids, cmd_attention_mask, text_ids, text_attention_mask, labels, prefix_lengths, text_lengths, classification_labels = batch
                out = gpt2_model(cmd_ids, attention_mask=cmd_attention_mask)
                cmd_past = out.past_key_values
                loss, acc, n_loss = model(cmd_past, text_ids, cmd_attention_mask, text_attention_mask, prefix_lengths, text_lengths, labels)
            elif args.task.startswith('oracle'):
                input_ids, attention_mask, length_labels, classification_labels = batch
                labels = length_labels if args.task == 'oracle_length' else classification_labels # oracle_label
                loss, acc, n_loss = model(input_ids, attention_mask, labels)
            else:
                raise NotImplementedError
            loss_meter.update(loss.detach(), int(n_loss))
            acc_meter.update(acc.detach(), int(n_loss))
            if batch_num % args.train_print_freq == 0:
                progress.display(batch_num)
                wandb.log({
                    'batch_val_loss': loss_meter.val, 
                    'batch_val_loss_avg': loss_meter.avg,
                    'batch_val_acc_avg': acc_meter.avg
                })
    progress.display(total_length)
    if per_class_acc:
        print(f'position,acc')
        for i in range(len(acc_pos_meters)):
            print(f'{i*5},{acc_pos_meters[i].avg}')
    wandb.log({'epoch_val_loss': loss_meter.avg, 'epoch_val_acc': acc_meter.avg, 'epoch': epoch})
    return loss_meter.avg

def load_model_no_state(args, tokenizer):
    if args.task == 'concat' or args.task == 'concatoracle':
        model_str = args.concat_model_path if args.concat_model_path is not None else LANGUAGE_MODEL_STRING
        gpt2_model = AutoModelForCausalLM.from_pretrained(model_str, pad_token_id=tokenizer.encode(PAD_TOKEN)[0])
        gpt2_model.resize_token_embeddings(len(tokenizer)) # because we added [PAD] as special token
        return ConcatModel(gpt2_model)
    elif args.task == 'concatfreeze':
        gpt2_model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_STRING, pad_token_id=tokenizer.encode(PAD_TOKEN)[0])
        gpt2_model.resize_token_embeddings(len(tokenizer)) # because we added [PAD] as special token
        change_position_ids = args.debug_freeze_pos_ids
        return FreezeConcatModel(gpt2_model, change_position_ids)
    elif args.task == 'alignment':
        return CommandAlignmentClassifier(tokenizer, classifier_dropout=CLASSIFIER_DROPOUT)
    elif args.task.startswith('oracle'):
        if args.task == 'oracle_label': oracle_num_classes = args.label_num_classes
        else: oracle_num_classes = args.length_num_classes
        if args.multiclass:
            return OracleClassifier(tokenizer, oracle_num_classes, CLASSIFIER_DROPOUT)
        else:
            return OracleBinaryClassifier(tokenizer, oracle_num_classes, CLASSIFIER_DROPOUT)
    elif args.task == 'alignfreeze':
        change_position_ids = args.debug_freeze_pos_ids
        return FreezeCommandAlignmentClassifier(tokenizer, classifier_dropout=CLASSIFIER_DROPOUT, change_position_ids=change_position_ids)
    else:
        raise NotImplementedError

def select_loss_function(args):
    if args.task == 'concat':
        return nn.CrossEntropyLoss(reduction='none')
    else:
        return nn.BCEWithLogitsLoss().to(args.device)

def main(args):
    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        model = load_model_no_state(args, dataset.tokenizer)
        if args.task == 'concat':
            state_dict = {k[7:]:v for k,v in checkpoint['state_dict'].items()} # fix 'module.model.xxx' error (dump 'module.' prefix in state dict keys)
        else:
            state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        data_start_index = checkpoint['data_start_index']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
        # NOTE: just import pdb after loading the model here if you want to play with it, it's easy
        # model.eval()
        # import pdb; pdb.set_trace()
    else:
        model = load_model_no_state(args, dataset.tokenizer)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.scheduler_type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma) if args.schedule else None
        if args.scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.gamma, total_iters=args.epochs) if args.schedule else None
        if args.scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(dataset.splits['train'])//args.batch_size) if args.schedule else None
        best_val_metric = 1e8 # lower is better for BCE
        data_start_index = 0
    if not args.no_dataparallel: model = nn.DataParallel(model) # multi-GPU
    print(f'GPU count: {torch.cuda.device_count()}')
    print('num params', num_params(model))
    criterion = select_loss_function(args)
    if args.task.endswith('freeze'):
        if args.debug_freeze_model:
            # for debugging, use the same model (essentially does not freeze)
            gpt2_model = model.model
        else:
            gpt2_model = AutoModel.from_pretrained('gpt2')
            gpt2_model = gpt2_model.to(args.device)
            gpt2_model.eval()
    else:
        gpt2_model = None
    
    if args.evaluate:
        epoch = 0
        validate(model, dataset, criterion, epoch, args, True, gpt2_model)
        return
    for epoch in range(args.epochs):
        print("TRAINING: Epoch {} at {}".format(epoch, time.ctime()))
        data_start_index = train(model, dataset, optimizer, scheduler, epoch, args, data_start_index, gpt2_model)
        if epoch % args.validation_freq == 0:
            print("VALIDATION: Epoch {} at {}".format(epoch, time.ctime()))
            metric = validate(model, dataset, criterion, epoch, args, False, gpt2_model)

            if not args.debug:
                if metric < best_val_metric:
                    print('new best val metric', metric)
                    best_val_metric = metric
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_metric': best_val_metric,
                        'optimizer': optimizer.state_dict(),
                        'data_start_index': data_start_index,
                        'args': args
                    }, os.path.join(args.save_dir, 'model_best.pth.tar'))
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': metric,
                    'optimizer': optimizer.state_dict(),
                    'data_start_index': data_start_index,
                    'args': args
                }, os.path.join(args.save_dir, 'model_epoch' + str(epoch) + '.pth.tar'))


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--task', type=str, required=True, choices=['concat', 'alignment', 'oracle_length', 'oracle_label', 'multi_oracle_length', 'multi_oracle_label', 'alignfreeze', 'concatfreeze', 'concatoracle'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--template_file', type=str, required=True)
    parser.add_argument('--use_content_form', action='store_true')
    parser.add_argument('--use_length', action='store_true')
    parser.add_argument('--machine_gen', type=str, default=None, help='path to machine generated file for negative example')
    parser.add_argument('--extra_data', type=str, nargs='*', default=[], help='extra data, argument is pair of (hf dataset, template); only used in train, not val or test; #args is even')

    # MODEL
    parser.add_argument('--multiclass', action='store_true', help='use OracleClassifier instead of OracleBinaryClassifier')
    parser.add_argument('--label_num_classes', type=int, default=5, help='for concatoracle and oracle')
    parser.add_argument('--length_num_classes', type=int, default=5, help='for concatoracle and oracle')

    # SAVE/LOAD
    parser.add_argument('--save_dir', type=str, required=True, help='where to save ckpts')
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')

    # TRAINING
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--epoch_max_len', type=int, default=None, help='max batches per epoch if set, for more frequent validation')
    parser.add_argument('--validation_freq', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_workers', type=int, default=20, help='num workers for data loader')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--no_dataparallel', action='store_true')
    parser.add_argument('--clip', type=float, default=None, help='gradient clipping')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--schedule', action='store_true', help='use scheduler')
    parser.add_argument('--scheduler_type', type=str, choices=['linear', 'exp', 'onecycle'], default='exp')
    parser.add_argument('--gamma', type=float, default=0.9, help='scheduler gamma')

    # ZERO-SHOT
    parser.add_argument('--block_label_class', type=int, nargs='*', default=[])
    parser.add_argument('--block_length_class', type=int, nargs='*', default=[])
    parser.add_argument('--block_completely', action='store_true', help='if true, do not even show data for zero-shot class (zero-shot 1). if false, show data but do not show command describing that class (zero-shot 2)')

    # COMPOSITIONAL
    parser.add_argument('--noncomp_label_class', type=int, nargs='*', default=[])

    # unblock concat base
    parser.add_argument('--concat_model_path', type=str, default=None)

    # PRINTING
    parser.add_argument('--train_print_freq', type=int, default=200, help='how often to print metrics (every X batches)')

    # DEBUG
    parser.add_argument('--debug_freeze_model', action='store_true', help='does not really freeze, to debug freeze model')
    parser.add_argument('--debug_freeze_pos_ids', action='store_true', help='if turned on, DOES change position_ids, otherwise DOES NOT change position_ids')

    parser.add_argument('--few_shot', action='store_true', help='only for wandb logging purposes')

    args = parser.parse_args()
    assert len(args.extra_data) % 2 == 0
    # if args.task == 'alignfreeze':
    #     assert torch.cuda.device_count() >= 2, 'need at least two gpus'
    assert not os.path.exists(args.save_dir) or len(os.listdir(args.save_dir)) == 0, 'PATH CONFLICT WITH EXISTING CHECKPOINTS! stopping...'

    # multiclass oracle: hack the args to avoid the need to pass in --multiclass
    if args.task.startswith('multi_'):
        args.task = args.task[6:] # multi_oracle_xxx -> oracle_xxx
        args.multiclass = True

    print(f'All args:{args}\n\n')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        assert args.ckpt is not None
    if len(args.block_label_class) > 0 or len(args.block_length_class) > 0:
        task_prefix = 'zero-shot'
    else: 
        task_prefix = args.task.split('_')[0]
    if args.few_shot: task_prefix = 'few-shot'
    wandb.init(
        mode = 'disabled' if args.evaluate else 'offline',
        project=f'task_{task_prefix}',
        name=args.save_dir,
        config={
            'args': args
        }
    )
    main(args)
    wandb.finish()