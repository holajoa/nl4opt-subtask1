import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from log import logger
from utils.reader_utils import get_ner_reader, extract_spans, _assign_ner_tags

from typing import List, Tuple, Dict
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CoNLLReader(Dataset):
    def __init__(self, max_instances=-1, max_length=50, target_vocab=None, pretrained_dir='', encoder_model='xlm-roberta-large'):
        self._max_instances = max_instances    # maximum number of training examples
        self._max_length = max_length    # maximum input sequence length
        # Initialise tokeniser
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)    
        # get the special tokens 
        self.pad_token = self.tokenizer.special_tokens_map['pad_token']    # pad token (a string, eg: '<pad>')
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]    # id of the pad token
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']    # sep token (eos token, eg: '</s>' here)

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []    # each element will contain five objects: subtoken_ids, masks, subtoken_masks, spans:Dict, tag_ids
        self.sentences = []

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):    # call get_ner_reader on file to read data
        """
        Adds instance from data to the Reader. Each instance is a tuple of five tensors/dicts:
            1. tokens_tensor: contains subtoken ids 
            2. mask_rep: contains masks (for what?? seems to be all true)
            3. token_masks_rep: masks for start-of-words tokens
            4. gold_spans_: dict that contains spans (subtoken level)
            5. tag_tensor: subtoken tag ids
        """
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):    # get_ner_reader() returns generator
            if self._max_instances != -1 and instance_idx > self._max_instances:  # Only get the first self._max_instances examples
                break
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask = self.parse_line_for_ner(fields=fields)
            
            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)    # convert subtoken ids to torch long tensor
            tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)    # convert subtoken tag ids to torch long tensor - reshape into column vector
            token_masks_rep = torch.tensor(token_masks_rep)    # convert subtoken masks to torch long tensor (True if subtoken is the start of word)
            mask_rep = torch.tensor(mask)    # convert masks to tensor

            self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor))
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields:List[List[str]]) -> Tuple[str, List[int], List[bool], Dict[tuple, str]]:
        """
        In addition to the results from parse_tokens_for_ner, extract spans and convert tags to ids.
        ------------------------------------------------------------------------------------------------
        
        :input:fields:List of lists:
            - First list: strings of full words
            - Last list: tags

        :return:sentence_str: string of subtokens joined by ' ', eg:' ▁ca uti ous ▁ asset ▁investment'
        :return:tokens_sub_rep: List of id of subtokens eg: [12, 12, 4, 5, 5, 12, 12, 12]
        :return:token_masks_rep: List of boolean values, indicating whether the current token is the start of a word. 
                                 Useful for tagging later.  eg: [True, False, False, False, True, True]
        :return:coded_ner_: List of tag ids of each token.
        :return:gold_spans_: Dictionary, spans extracted from subtokens. Keys are the span positions and vals are the tags. 
        :return:mask: List of bools, whether current position is a token (not a special token)
        """
        tokens_, ner_tags = fields[0], fields[-1]   # extract the words and tags (two discarded lists are underscores)
        self.sentences.append(tokens_)
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep)    # Extract spans from the tags
        # Convert the list of tags to list of tag ids: coded_ner_
        coded_ner_ = [self.label_to_id[tag] if tag in self.label_to_id else self.label_to_id['O'] for tag in ner_tags_rep]    # unrecoded tags are treated as O tag
        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask

    def parse_tokens_for_ner(self, tokens_:List[str], ner_tags:List[str]):
        """
        Parse a sentence. 

        :input:tokens_: List of strings, each containing one token (full word)
        :input:ner_tags: List of tags.

        :return:sentence_str: string of subtokens joined by ' ', eg:' ▁ca uti ous ▁ asset ▁investment'
        :return:tokens_sub_rep: List of id of subtokens eg: [12, 12, 4, 5, 5, 12, 12, 12]
        :return:ner_tags_rep: List of tags of SUBtokens - one tag for each SUBtoken. 
                            May need to change B- and I- tags since the starting subtoken may change - done in _assign_ner_tags().
        :return:token_masks_rep: List of boolean values, indicating whether the current token is the start of a word. 
                                 Useful for tagging later.  eg: [True, False, False, False, True, True]
        :return:mask: List of bools (?) all true, including the ending pad token
        """
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        token_masks_rep = [False]
        ### Each iteration deal with one FULL WORD
        for idx, token in enumerate(tokens_):
            ### ========================= Dealing with tokens =========================
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:   # truncate sequences that are too long
                break
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))  # string of post-processed sub-tokens, eg: ' ▁ca uti ous'
            rep_ = self.tokenizer(token.lower())['input_ids']    # get input id of subtokens
            rep_ = rep_[1:-1]    # strip the start and ending token <s> and </s> that the tokeniser automatically adds onto the token 
            tokens_sub_rep.extend(rep_)    # append the ids to the sentence-level string

            ### ========================= Dealing with tags: see docs of _assign_ner_tags =========================
            # When the full word's tag starts with B, the tag of its first subtoken should still start with B, 
            # while the succeeding subtokens should have tags starting with I now.
            ner_tag = ner_tags[idx]    # tag of full word
            tags, masks = _assign_ner_tags(ner_tag, rep_)    # get the tags for subtokens and SOW masks (start-of-word masks)
            # Append tags and masks to the sentence-level lists
            ner_tags_rep.extend(tags)
            token_masks_rep.extend(masks)

        # Append one <pad> token at the end of the text to indicate ending - this is manifested in the list of ids, tags and masks (WHYYYY ??)
        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep.append(False)
        mask = [True] * len(tokens_sub_rep)
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask
