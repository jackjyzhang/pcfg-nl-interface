import torch
import torch.nn as nn 
from transformers import AutoModel
from constants import LANGUAGE_MODEL_STRING
from util import prefix_and_pad_mask
class ConcatModel(nn.Module):
    def __init__(self, model):
        super(ConcatModel, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, input_ids, attention_masks, loss_masks):
        out = self.model(input_ids, attention_mask=attention_masks)

        shift_logits = out.logits[..., :-1, :] # bz x (seqlen-1) x vocab_size
        shift_labels = input_ids[..., 1:] # bz x (seqlen-1)
        shift_loss_masks = loss_masks[..., 1:] # bz x (seqlen-1)
        losses = self.criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)) # bz*(seqlen-1) 1D
        n_loss = shift_loss_masks.sum()
        loss = (losses * shift_loss_masks.reshape(-1)).sum() / n_loss

        return loss, n_loss

class FreezeConcatModel(nn.Module):
    def __init__(self, model, change_position_ids=False):
        super(FreezeConcatModel, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.change_position_ids = change_position_ids
    
    def forward(self, cmd_past, text_ids, cmd_attention_mask, text_attention_mask):
        bz, inputlen = cmd_attention_mask.size(0), text_ids.size(1)
        attention_mask = torch.cat([cmd_attention_mask, text_attention_mask], dim=1).contiguous()
        prefix_lengths = cmd_attention_mask.sum(dim=1)
        position_ids = torch.arange(inputlen, device=text_ids.device).unsqueeze(0).expand(bz,inputlen) + prefix_lengths.unsqueeze(1) if self.change_position_ids else None

        out = self.model(text_ids, attention_mask=attention_mask, past_key_values=cmd_past, position_ids=position_ids) # logits (bz,inputlen,vocab)
        loss_mask = (text_attention_mask == 1).contiguous()

        shift_logits = out.logits[..., :-1, :] # bz x (inputlen-1) x vocab_size
        shift_labels = text_ids[..., 1:].contiguous() # bz x (inputlen-1)
        shift_loss_masks = loss_mask[..., 1:] # bz x (input-1)
        flat_shift_loss_masks = shift_loss_masks.flatten()
        flat_masked_logits = shift_logits.flatten(0, 1)[flat_shift_loss_masks]
        flat_masked_labels = shift_labels.flatten()[flat_shift_loss_masks]
        loss = self.criterion(flat_masked_logits, flat_masked_labels)
        n_loss = shift_loss_masks.sum()

        return loss, n_loss

class CommandAlignmentClassifier(nn.Module):
    def __init__(self, tokenizer, classifier_dropout=0.1):
        super(CommandAlignmentClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(LANGUAGE_MODEL_STRING)
        self.model.resize_token_embeddings(len(tokenizer)) # [BOS], [PAD], [SEP] are added
        self.criterion = nn.BCEWithLogitsLoss()
        self.hidden_size = self.model.config.n_embd
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, 1)

    def _run_layers(self, input_ids, attention_mask, past=None, get_past=False):
        out = self.model(input_ids, attention_mask=attention_mask, past_key_values=past) # out.last_hidden_state (bz,seqlen,hs)
        last_hidden_state = self.dropout(out.last_hidden_state) # (bz,seqlen,hs)
        logits = self.classifier(last_hidden_state).squeeze(2) # (bz,seqlen)
        if get_past:
            return logits, out.past_key_values
        else:
            return logits

    def forward(self, input_ids, attention_mask, prefix_lengths=None, text_lengths=None, labels=None, per_class_acc=False):
        '''
        input_ids: command and input text concatenated together (but separated by a separated), padded -  (bz,seqlen)
        attention_masks: (bz,seqlen)
        labels: 0/1 labels indicating whether command & input align or not - (bz,)
        '''
        logits = self._run_layers(input_ids, attention_mask)

        if labels is not None: # for training, return loss
            expanded_labels = labels.unsqueeze(1).expand(-1, logits.size(1)) # (bz,seqlen) because we are learning for all positions simultaneously
            loss_mask = prefix_and_pad_mask(prefix_lengths, text_lengths) # (bz,seqlen)
            flat_loss_mask = loss_mask.flatten()
            flat_logits = logits.flatten()
            flat_expanded_labels = expanded_labels.flatten()
            masked_flat_logits = flat_logits[flat_loss_mask]
            masked_flat_expanded_labels = flat_expanded_labels[flat_loss_mask]
            loss = self.criterion(masked_flat_logits, masked_flat_expanded_labels.float())
            predicted_labels = (flat_logits >= 0.5).long()
            masked_predicted_labels = predicted_labels[flat_loss_mask]
            acc = (masked_predicted_labels == masked_flat_expanded_labels).float().mean()
            n = loss_mask.sum()
            if per_class_acc:
                acc0 = (masked_predicted_labels[masked_flat_expanded_labels==0] == masked_flat_expanded_labels[masked_flat_expanded_labels==0]).float().mean()
                acc1 = (masked_predicted_labels[masked_flat_expanded_labels==1] == masked_flat_expanded_labels[masked_flat_expanded_labels==1]).float().mean()
                n0 = (masked_flat_expanded_labels==0).sum()
                n1 = (masked_flat_expanded_labels==1).sum()
                return loss, acc, n, acc0, acc1, n0, n1, predicted_labels.detach(), flat_expanded_labels.detach()
            return loss, acc, n
        else: # for inference, return logits for the last seq position (when doing inference the last position should always be a text position!)
            return logits[:, -1] # (bz,)

    def inference(self, input_ids, attention_mask, past):
        '''
        Efficient inference using past key values
        '''
        logits, past = self._run_layers(input_ids, attention_mask, past, get_past=True)
        return logits[:, -1], past

class FreezeCommandAlignmentClassifier(nn.Module):
    def __init__(self, tokenizer, classifier_dropout=0.1, change_position_ids=False):
        super(FreezeCommandAlignmentClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(LANGUAGE_MODEL_STRING)
        self.model.resize_token_embeddings(len(tokenizer)) # [BOS], [PAD], [SEP] are added
        self.criterion = nn.BCEWithLogitsLoss()
        self.hidden_size = self.model.config.n_embd
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.change_position_ids = change_position_ids

    def _run_layers(self, cmd_past, text_ids, attention_mask, prefix_lengths, past=None, get_past=False):
        if past is None:
            bz, seqlen = attention_mask.size(0), attention_mask.size(1)
            inputlen = text_ids.size(1)
            # position_ids = torch.cat([
            #     torch.arange(seqlen-inputlen, device=text_ids.device).unsqueeze(0).expand(bz,seqlen-inputlen), # cmd part
            #     torch.arange(inputlen, device=text_ids.device).unsqueeze(0).expand(bz,inputlen) + prefix_lengths.unsqueeze(1) # text part
            # ], dim=1)
            position_ids = torch.arange(inputlen, device=text_ids.device).unsqueeze(0).expand(bz,inputlen) + prefix_lengths.unsqueeze(1) if self.change_position_ids else None
            out = self.model(text_ids, attention_mask=attention_mask, past_key_values=cmd_past, position_ids=position_ids) # out.last_hidden_state (bz,seqlen,hs)
        else:
            out = self.model(text_ids, attention_mask=attention_mask, past_key_values=past)
        last_hidden_state = self.dropout(out.last_hidden_state) # (bz,seqlen,hs)
        logits = self.classifier(last_hidden_state).squeeze(2) # (bz,seqlen)
        if get_past:
            return logits, out.past_key_values
        else:
            return logits

    def forward(self, cmd_past, text_ids, cmd_attention_mask, text_attention_mask, prefix_lengths, text_lengths, labels):
        '''
        input_ids: command and input text concatenated together (but separated by a separated), padded -  (bz,seqlen)
        attention_masks: (bz,seqlen)
        labels: 0/1 labels indicating whether command & input align or not - (bz,)
        '''
        attention_mask = torch.cat([cmd_attention_mask, text_attention_mask], dim=1).contiguous()
        logits = self._run_layers(cmd_past, text_ids, attention_mask, prefix_lengths)

        expanded_labels = labels.unsqueeze(1).expand(-1, logits.size(1)) # (bz,seqlen) because we are learning for all positions simultaneously
        # loss_mask = torch.cat([torch.zeros_like(cmd_attention_mask), text_attention_mask], dim=1).contiguous() == 1 # (bz,seqlen), mask all command positions (a square-shaped prefix)
        loss_mask = (text_attention_mask == 1).contiguous()
        flat_loss_mask = loss_mask.flatten()
        flat_logits = logits.flatten()
        flat_expanded_labels = expanded_labels.flatten()
        masked_flat_logits = flat_logits[flat_loss_mask]
        masked_flat_expanded_labels = flat_expanded_labels[flat_loss_mask]
        loss = self.criterion(masked_flat_logits, masked_flat_expanded_labels.float())
        predicted_labels = (flat_logits >= 0.5).long()
        masked_predicted_labels = predicted_labels[flat_loss_mask]
        acc = (masked_predicted_labels == masked_flat_expanded_labels).float().mean()
        n = loss_mask.sum()

        return loss, acc, n


    def inference(self, cmd_past, text_ids, attention_mask, prefix_lengths, past):
        '''
        Efficient inference using past key values
        '''
        logits, past = self._run_layers(cmd_past, text_ids, attention_mask, prefix_lengths, past, get_past=True)
        return logits[:, -1], past

class OracleClassifier(nn.Module):
    def __init__(self, tokenizer, n_classes, classifier_dropout=0.1):
        super(OracleClassifier, self).__init__()
        self.n_classes = n_classes
        self.model = AutoModel.from_pretrained(LANGUAGE_MODEL_STRING)
        self.model.resize_token_embeddings(len(tokenizer)) # [BOS], [PAD], [SEP] are added
        self.criterion = nn.CrossEntropyLoss()
        self.hidden_size = self.model.config.n_embd
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.n_classes)

    def _run_layers(self, input_ids, attention_mask, past=None, get_past=False):
        out = self.model(input_ids, attention_mask=attention_mask, past_key_values=past) # out.last_hidden_state (bz,seqlen,hs)
        last_hidden_state = self.dropout(out.last_hidden_state) # (bz,seqlen,hs)
        logits = self.classifier(last_hidden_state).squeeze(2) # (bz,seqlen,n_classes)
        if get_past:
            return logits, out.past_key_values
        else:
            return logits

    def forward(self, input_ids, attention_mask, labels):
        '''
        input_ids: command and input text concatenated together (but separated by a separated), padded -  (bz,seqlen)
        attention_masks: (bz,seqlen)
        labels: 0/1 labels indicating whether command & input align or not - (bz,)
        '''
        logits = self._run_layers(input_ids, attention_mask)

        expanded_labels = labels.unsqueeze(1).expand(input_ids.size()).contiguous() # (bz,seqlen) because we are learning for all positions simultaneously
        loss_mask = (attention_mask == 1).contiguous() # (bz,seqlen)
        flat_loss_mask = loss_mask.view(-1) # (bz*seqlen)
        flat_logits = logits.view(-1, self.n_classes) # (bz*seqlen,n_classes)
        flat_expanded_labels = expanded_labels.view(-1) # (bz*seqlen)
        masked_flat_logits = flat_logits[flat_loss_mask]
        masked_flat_expanded_labels = flat_expanded_labels[flat_loss_mask]
        loss = self.criterion(masked_flat_logits, masked_flat_expanded_labels)
        n_loss = loss_mask.sum()
        _, flat_predicted_labels = flat_logits.max(axis=1) # (bz*seqlen,n_classes) -> (bz*seqlen)
        flat_masked_predicted_labels = flat_predicted_labels[flat_loss_mask]
        acc = (flat_masked_predicted_labels == masked_flat_expanded_labels).float().mean()
        # masked positions are padding, so they should still have high acc! so if acc use the non-masked version it could go higher?
            
        return loss, acc, n_loss


    def inference(self, input_ids, attention_mask, past):
        '''
        Efficient inference using past key values
        '''
        logits, past = self._run_layers(input_ids, attention_mask, past, get_past=True)
        return logits, past # just return the full logits (easier for future code)

class OracleBinaryClassifier(nn.Module):
    def __init__(self, tokenizer, n_classes, classifier_dropout=0.1):
        super(OracleBinaryClassifier, self).__init__()
        self.n_classes = n_classes
        self.model = AutoModel.from_pretrained(LANGUAGE_MODEL_STRING)
        self.model.resize_token_embeddings(len(tokenizer)) # [BOS], [PAD], [SEP] are added
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.hidden_size = self.model.config.n_embd
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifiers = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(self.n_classes)])

    def _run_layers(self, input_ids, attention_mask, past=None, get_past=False):
        out = self.model(input_ids, attention_mask=attention_mask, past_key_values=past) # out.last_hidden_state (bz,seqlen,hs)
        last_hidden_state = self.dropout(out.last_hidden_state) # (bz,seqlen,hs)
        logits = [cls_head(last_hidden_state).squeeze().unsqueeze(1) for cls_head in self.classifiers] # (bz,seqlen,1) -> (bz,1,seqlen) list of length n_classes
        logits = torch.cat(logits, dim=1).contiguous() # (bz,n_classes,seqlen)
        if get_past:
            return logits, out.past_key_values
        else:
            return logits

    def forward(self, input_ids, attention_mask, labels):
        '''
        input_ids: command and input text concatenated together (but separated by a separated), padded -  (bz,seqlen)
        attention_masks: (bz,seqlen)
        labels: 0/1 labels indicating whether command & input align or not - (bz,)
        '''
        logits = self._run_layers(input_ids, attention_mask) # (bz,n_classes,seqlen)
        bz,n_classes,seqlen = logits.size()
        flat_logits = logits.view(bz*n_classes,seqlen) # (bz*n_classes,seqlen)
        flat_binary_labels = (labels.unsqueeze(1).expand(bz,n_classes) == torch.arange(n_classes, device=labels.get_device()).unsqueeze(0).expand(bz,n_classes)).long().flatten() # (bz,n_classes) -> (bz*n_classes)
        flat_expanded_binary_labels = flat_binary_labels.unsqueeze(1).expand(-1,seqlen) # (bz*n_classes,seqlen)
        weight = torch.where(
            flat_binary_labels == 1,
            torch.ones(bz*n_classes, device=flat_binary_labels.get_device()),
            torch.ones(bz*n_classes, device=flat_binary_labels.get_device())/(n_classes-1)
        ).unsqueeze(1) # (bz*n_classes,1)
        # __import__('pdb').set_trace()
        loss_mask = (attention_mask == 1).unsqueeze(1).expand(logits.size()).contiguous() # (bz,seqlen) -> (bz,1,seqlen) -> (bz,n_classes,seqlen)
        flat_loss_mask = loss_mask.view(-1,loss_mask.size(2)) # (bz*n_classes,seqlen)
        unreduced_loss = self.criterion(flat_logits, flat_expanded_binary_labels.float()) * weight 
        loss = unreduced_loss[flat_loss_mask].mean()
        n_loss = loss_mask.sum()

        flat_predicted_labels = (flat_logits >= 0.5).long() # (bz*n_classes,seqlen)
        acc = (flat_predicted_labels[flat_loss_mask] == flat_expanded_binary_labels[flat_loss_mask]).float().mean()
        # masked positions are padding, so they should still have high acc! so if acc use the non-masked version it could go higher?
            
        return loss, acc, n_loss


    def inference(self, input_ids, attention_mask, past):
        '''
        Efficient inference using past key values
        '''
        logits, past = self._run_layers(input_ids, attention_mask, past, get_past=True)
        return logits, past # just return the full logits (easier for future code)
