'''
Load a template file and create relevant command
'''

import sys
import argparse
import random
import re
import copy
import time
import numpy as np
from scipy.stats import binom

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

organize_ws = lambda my_str: _RE_COMBINE_WHITESPACE.sub(" ", my_str).strip()

from nltk import CFG
from nltk.parse.generate import generate

from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template-path', '-t', type=str, required=True, help='path to template file')
    parser.add_argument('--dataset-path', '-d', type=str, required=True, help='path to huggingface dataset')
    parser.add_argument('--save-path', '-s', type=str, default='template-out.txt', help='path to output file')
    return parser.parse_args()

class Template:
    def __init__(self, template_file, tokenizer=None):
        self.var_transforms, self.templates, self.len_transforms, self.lab_transforms, self.label_name = self.load_template(template_file)
        self.NER_CLASS = ['EVENT, FAC, GPE, LANGUAGE, LAW, LOC, ORG, PERSON, PRODUCT, WORK_OF_ART']
        self.KW_CLASS = ['NOUN', 'PROPN']
        self.puncts_list = ['.', '?', '!', ',', '-'] # business -related -> business-related
        # the single brackets [SENT take care of [SENT-SEG] problem
        self.templates_no_content_form = [template for template in self.templates if template.find('[CONTENT-FORM') == -1]
        self.templates_no_length = [template for template in self.templates if template.find('[LENGTH') == -1]
        self.templates_no_content_and_length = [
            template for template in self.templates if (template.find('[CONTENT-FORM') == -1 and template.find('[LENGTH') == -1)
        ]
        self.templates_no_label = [template for template in self.templates if template.find(self.label_name[:-1]) == -1]
        self.templates_no_content_and_label = [
            template for template in self.templates if (template.find('[CONTENT-FORM') == -1 and template.find(self.label_name[:-1]) == -1)
        ]
        self.tokenizer = tokenizer
        assert len(self.templates_no_content_form) > 0
        assert len(self.templates_no_length) > 0
        assert len(self.templates_no_content_and_length) > 0

    @classmethod
    def load_template(cls, load_path):
        var_transforms = {}
        len_transforms = []
        lab_transforms = {}
        templates = []
        var_mode = True
        len_mode = False
        lab_mode = False
        tem_mode = False
        with open(load_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split('#')[0] # process comments
                if len(line) == 0: continue
                if line == '<variables>': var_mode=True; len_mode=False; lab_mode = False; tem_mode=False; continue
                if line == '<length>': var_mode=False; len_mode=True; lab_mode = False; tem_mode=False; continue
                if line.startswith('<label>'): 
                    var_mode=False; len_mode=False; lab_mode = True; tem_mode=False
                    label_name = line.split('\t')[1].strip()
                    continue
                if line == '<templates>': var_mode=False; len_mode=False; lab_mode = False; tem_mode=True; continue
                if var_mode:
                    # variable mode
                    parts = line.split('\t')
                    if len(parts) != 2:
                        continue
                    k, v = parts[0], parts[1].split(',')
                    values = []
                    for word in v:
                        word = word.strip()
                        word = word.replace('/', ',') # using '/' as escaped ','
                        values.append(word)
                    var_transforms[k] = values
                if len_mode:
                    head, tail = line.split('\t')
                    head = int(head)
                    tail = tail.split(',')
                    tail = [word.strip() for word in tail]
                    len_transforms.append([head, tail])
                if lab_mode:
                    head, tail = line.split('\t')
                    head = int(head)
                    tail = tail.split(',')
                    tail = [word.strip() for word in tail]
                    lab_transforms[head] = tail
                if tem_mode:
                    # template mode
                    if line.find('[') == -1:
                        continue # must have at least one pair of '[...]' in order to be a template
                    templates.append(line)
            # make sure len_transforms is ordered
            len_transforms = list(sorted(len_transforms, key=lambda x:x[0]))
            return var_transforms, templates, len_transforms, lab_transforms, label_name

    @classmethod
    def save_template(cls, save_path, var_transforms, templates, len_transforms, lab_transforms, label_name):
        with open(save_path, 'w', encoding='UTF-8') as f:
            # variables
            print('<variables>', file=f)
            for lhs, rhs_list in var_transforms.items():
                rhs_list = [word.replace(',', '/') for word in rhs_list]
                rhs = ', '.join(rhs_list)
                print(f'{lhs}\t{rhs}', file=f)
            # length
            print('\n<length>', file=f)
            for head, tail_list in len_transforms:
                tail = ', '.join(tail_list)
                print(f'{head}\t{tail}', file=f)
            # label
            print(f'\n<label>\t{label_name}', file=f)
            for head, tail_list in lab_transforms.items():
                tail = ', '.join(tail_list)
                print(f'{head}\t{tail}', file=f)
            # templates
            print('\n<templates>', file=f)
            for template in templates:
                print(template, file=f)

    def preprocess_rhs(self, rhs_str):
        parts = rhs_str.split(' ')
        for i in range(len(parts)):
            if not parts[i].startswith('['):
                parts[i] = f"'{parts[i]}'"
        return ' '.join(parts)

    def get_len_tags(self, text, negate=False):
        # with a tokenizer, get the encoded length. otherwise, get string length
        if self.tokenizer is not None:
            l = len(self.tokenizer.encode(text))
        else:
            l = len(text)
        for target_len, wordslist in self.len_transforms:
            # target_len is increasing. the last target_len is maxlen in dataset
            if l <= target_len:
                break
        if negate:
            all_neg_wordslist = []
            for tl, wl in self.len_transforms:
                if tl != target_len: all_neg_wordslist += wl
            return copy.deepcopy(all_neg_wordslist)
        else:
            return copy.deepcopy(wordslist)

    def get_label_tags(self, label, negate=False):
        if negate:
            all_neg_words = []
            for k,v in self.lab_transforms.items():
                if label != k: all_neg_words += v
            return copy.deepcopy(all_neg_words)
        else:
            return copy.deepcopy(self.lab_transforms[label])

    def _augment_transforms(self, example, add_templates=False, structurize=False, negate=False):
        '''
        Combine variable transforms & attribute transforms
        This is to prepare input for command generation
        '''
        aug_transforms = copy.deepcopy(self.var_transforms)
        if negate:
            N_ATTR = 2 # currently only considering 2 attributes: length and label
            # negate at least one of the attributes
            # binomial distribution with 0 blacked out
            num_to_negate = random.choices(range(1,N_ATTR+1), weights=[binom.pmf(i, N_ATTR, 0.5) for i in range(1,N_ATTR+1)])[0]
            flags = np.concatenate((np.ones(num_to_negate),np.zeros(N_ATTR-num_to_negate))).astype('bool').tolist() # True, ..., True, False, ..., False
            random.shuffle(flags)
            negate_length, negate_label = flags
        else:
            negate_length, negate_label = False, False

        aug_transforms['[LENGTH]'] = self.get_len_tags(example['text'], negate=negate_length)
        aug_transforms[self.label_name] = self.get_label_tags(example['label'], negate=negate_label)
        if add_templates:
            aug_transforms['[ROOT]'] = self.templates
        if structurize:
            for k in aug_transforms.keys():
                for i in range(len(aug_transforms[k])):
                    aug_transforms[k][i] = self._structurize_template(aug_transforms[k][i])
        return aug_transforms

    def _transform2grammar(self, transform):
        grammar = '[S] -> [ROOT]\n'
        for k,vlist in transform.items():
            for v in vlist:
                grammar += f"{k} -> {self.preprocess_rhs(v)}\n"
        return grammar.replace('[','').replace(']','')

    def generate_commands(self, example, file=sys.stdout):
        '''
        Recursively create all possible commands for a single example

        example must have at least one of 'entities', 'keywords' field
        '''
        aug_transforms = self._augment_transforms(example, add_templates=True, structurize=False)
        aug_transforms['[CONTENT-FORM]'] = self.sample_content_form(example)
        grm_str = self._transform2grammar(aug_transforms)
        grammar = CFG.fromstring(grm_str)
        for sentence in generate(grammar):
            print(' '.join(sentence), file=file)

    # --above is for generate all--

    def sample_one_content_form(self, N, ADJ, p_addadj, wt_wo_adj):
        if len(ADJ) > 0:
            if random.random() < p_addadj:
                # add ADJ to N
                num_adjs = random.choice(range(len(ADJ)))
                adjs = random.sample(ADJ, num_adjs)
                for adj in adjs:
                    assist_noun = random.choices(['stuff', 'topic', 'content', 'subject matter'], weights=[2,2,2,1])
                    N.append(f'{adj} {assist_noun}')

        # choose number of Ns
        num_N = random.choices(list(range(1,4))[:len(N)], weights=wt_wo_adj[:len(N)])[0]
        Ns = random.sample(N, num_N)
        if num_N == 1:
            return [Ns[0]]
        if num_N == 2:
            return [Ns[0], 'and', Ns[1]]
        if num_N == 3:
            return [Ns[0], ',',  Ns[1], 'and', Ns[2]]

    def sample_content_form(self, example, n_samples=100, p_addadj=0.3, wt_wo_adj=[3,2,1]):
        N = [text for text, tag, _ in example['entities'] if tag in self.NER_CLASS]
        ADJ = []
        for text, tag, _ in example['keywords']:
            if tag in self.KW_CLASS: N.append(text)
            if tag == 'ADJ': ADJ.append(text)
        if len(N) == 0:
            return [] # can't generate any content form because there are not attributes provided!
        return [self.sample_one_content_form(N, ADJ, p_addadj, wt_wo_adj) for _ in range(n_samples)]
            
    # --below is for generate one--

    def _is_nonterminal(self, text):
        return text.startswith('[')

    def _choose_rhs(self, transforms, lhs):
        '''
        lhs is a string
        return: rhs is a list
        '''
        rhs_list = transforms[lhs]
        return random.choice(rhs_list)

    def _dfs(self, node, transforms, output):
        for item in node:
            if self._is_nonterminal(item):
                rhs = self._choose_rhs(transforms, item)
                self._dfs(rhs, transforms, output)
            else:
                output.append(item)

    def _generate_one(self, template, transforms):
        '''
        Generate one sentence
            template: a single template to start with
            transforms: the CFG grammar
        '''
        node = template.copy()
        output = []
        self._dfs(node, transforms, output)
        return output

    def _structurize_template(self, template_str):
        if len(template_str) == 0:
            return []
        if template_str[-1] in self.puncts_list:
            punct = template_str[-1]
            template_str = template_str[:-1]
        else:
            punct = ''
        # assume string is well-formed
        parts = []
        while len(template_str) > 0:
            l = template_str.find('[')
            r = template_str.find(']')
            if l == -1:
                parts.append(template_str)
                break
            if l == 0:
                parts.append(template_str[:r+1])
                template_str = template_str[r+1:]
            else:
                # l > 0
                parts.append(template_str[:l])
                parts.append(template_str[l:r+1])
                template_str = template_str[r+1:]
        if len(punct) > 0:
            parts.append(punct)

        return [part.strip() for part in parts]
    
    def _postprocess(self, sent_list):
        '''
        0. delete 'the' after 'a', i.e. 'a the world related news' -> 'a world related news'
        1. fix 'a' before a vowel to 'an'
        2. 'please I need' => 'please , I need'
        3. capitalize first word in sentence
        4. delete whitespace between word and punctuation
        '''
        sent_list = [seg for seg in sent_list if len(seg) > 0] # filter out empty string
        sent_list = [ell for el in sent_list for ell in el.split(' ')]
        # 0.
        to_remove = []
        for i in range(1, len(sent_list)):
            if sent_list[i] == 'the' and sent_list[i-1] == 'a':
                to_remove.append(i)
        sent_list = [seg for i, seg in enumerate(sent_list) if not i in to_remove]
        # 1.
        for i in range(len(sent_list)-1):
            if sent_list[i] == 'a' and sent_list[i+1][0] in ['a','e','i', 'o', 'u']:
                sent_list[i] = 'an'
        # 2.
        if len(sent_list) >= 2 and sent_list[0] == 'please' and sent_list[1][0] == 'I':
            sent_list.insert(1, ',')
        # 3.
        sent_list[0] = sent_list[0].capitalize() # assume nonempty
        for i in range(1, len(sent_list)):
            if sent_list[i-1].endswith('.'):
                sent_list[i] = sent_list[i].capitalize()
        # 4.
        sent = []
        for i in range(1, len(sent_list)):
            if sent_list[i-1][0] in self.puncts_list:
                continue
            if sent_list[i][0] in self.puncts_list:
                sent.append(sent_list[i-1]+sent_list[i])
            else:
                sent.append(sent_list[i-1])
        assert sent_list[-1][0] in self.puncts_list, 'Sentence not ended with a punctuation!'
        return sent

    def generate_single(self, example, allow_content=True, allow_length=True, allow_label=True, return_metadata=False, negate=False, strict=False):
        '''
            main method to sample a single command

            negate: generate a negative command (command that does not describe the example text)
        '''
        aug_transforms = self._augment_transforms(example, add_templates=False, structurize=True, negate=negate)
        aug_transforms['[CONTENT-FORM]'] = self.sample_content_form(example, n_samples=1) if (allow_content and (not negate)) else [] # when sampling negative command, always not include content form (for now 0305)
        allow_content = len(aug_transforms['[CONTENT-FORM]']) > 0
        if allow_label:
            if not allow_content and allow_length:
                template_choices = self.templates_no_content_form
            elif allow_content and not allow_length:
                template_choices = self.templates_no_length
            elif not allow_content and not allow_length:
                template_choices = self.templates_no_content_and_length
            else: # allow_content and allow_length
                template_choices = self.templates
        else:
            assert allow_length # can't only have content (for now)
            if allow_content:
                template_choices = self.templates_no_label
            else:
                template_choices = self.templates_no_content_and_label
        if strict:
            strict_choices = []
            for template in template_choices:
                if (allow_content and template.find('[CONTENT-FORM') == -1) or (not allow_content and template.find('[CONTENT-FORM') != -1) : continue
                if (allow_length and template.find('[LENGTH') == -1) or (not allow_length and template.find('[LENGTH') != -1): continue
                if (allow_label and template.find(self.label_name[:-1]) == -1) or (not allow_label and template.find(self.label_name[:-1]) != -1): continue
                strict_choices.append(template)
            template_choices = strict_choices
        template_str = random.choice(template_choices)
        template = self._structurize_template(template_str)
        if return_metadata:
            has_form = False
            has_attr = False
            has_label = False
            has_length = False
            for part in template:
                if part == '[TEXT-FORM]': has_form = True
                if part == '[CONTENT-FORM]': has_attr = True
                if self.label_name[:-1] in part: has_label = True # fix [SENT-SEG] issue ([SENT-SEG] has [SENT as a prefix)
                if part == '[LENGTH]': has_length = True
        sent = self._generate_one(template, aug_transforms)
        sent = self._postprocess(sent)
        final_sent = organize_ws(' '.join(sent))
        if return_metadata:
            content_form_str = aug_transforms['[CONTENT-FORM]'][0] if len(aug_transforms['[CONTENT-FORM]']) > 0 else ''
            return final_sent, has_form, has_attr, has_label, has_length, content_form_str
        else:
            return final_sent
    
    def generate_per_template(self, example, n=1, file=None):
        '''
        Create n sentences for each template
        '''
        all_outputs = []
        aug_transforms = self._augment_transforms(example, add_templates=False, structurize=True)
        lt = len(self.templates)
        CONTENT_FORMS = self.sample_content_form(example, n_samples=n*lt)
        for t, template_str in enumerate(self.templates):
            template = self._structurize_template(template_str)
            for i in range(n):
                aug_transforms['[CONTENT-FORM]'] = [CONTENT_FORMS[n*t+i]]
                sent = self._generate_one(template, aug_transforms)
                sent = self._postprocess(sent)
                all_outputs.append(organize_ws(' '.join(sent)))
        
        if file is not None:
            for sent in all_outputs:
                print(sent, file=file)

        return all_outputs


if __name__ == '__main__':
    args = parse_args()
    template_obj = Template(args.template_path)
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset['train']
    random.seed(42)
    N = 100
    idxs = random.sample(range(len(dataset)), N)
    with open(args.save_path, 'w') as f:
        # yelp_template.generate_commands(ex, f)
        t1 = time.time()
        for i in idxs:
            ex = dataset[i]
            print(f'text: {ex["text"]}', file=f)
            print(f'label: {ex["label"]}', file=f)
            # template_obj.generate_per_template(ex, n=2, file=f)
            command_str = template_obj.generate_single(ex, allow_content=False, allow_length=True, allow_label=True, return_metadata=False, negate=False, strict=False)
            print(f'command: {command_str}', file=f)
            print('-'*70, file=f)
        t2 = time.time()
        print(f'average time per sample: {(t2-t1)/N} seconds')
