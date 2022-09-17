import gzip
import itertools
from typing import List, Dict


def get_ner_reader(data):
    """
    Parse the ner data into understandable format. Returns generator.
    
    return fields: four lists:
        [0]: List of strings, each string is a word. Word form the original consistent text. 
        [1] & [2]: List of underscores '_'. Length is the number of words.
        [3]: List of tags. One corresponds to one words
    return metadata: ?
    """
    fin = gzip.open(data, 'rt', encoding='UTF-8') if data.endswith('.gz') else open(data, 'rt', encoding='UTF-8')
    for is_divider, lines in itertools.groupby(fin, _is_divider):  # (True, {all lines that are divider (contains no info)}), (False, {all lines that are not divider (useful)})
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '').replace('\u200b', '') for line in lines]    # convert useful lines from iterator to list
        # Now each line in lines is of the form:'Cautious\t_\t_\tO'
        # ie: '[token]\t_\t_\t[tag]'

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None    # may have metadata provided - string should start with # id if so. 
        fields = [line.split() for line in lines if not line.startswith('# id')]    # each line is split into a list of strings, each sublist look like '['Cautious', '_', '_', 'O']'.
        fields = [list(field) for field in zip(*fields)]    # now fields is a list containing 4 sublists as specified in the docstring

        yield fields, metadata    # yield one sentence each iteration


def _assign_ner_tags(ner_tag, rep_):
    '''
    Changing the token_masks so that only the first sub_word of a token has a True value, 
    while the rest is False. This will be used for storing the predictions.
    
    :param ner_tag:
    :param rep_:
    :return ner_tags_rep:
    :return mask_:
    '''
    ner_tags_rep = []    # initialise list of subtoken tags

    sub_token_len = len(rep_)    # number of subtokens in the full word
    mask_ = [False] * sub_token_len    # initialise SOW masks
    # if the word is not empty, assign SOW mask to the first subtoken to be true
    if len(mask_):    
        mask_[0] = True
    # if the full word tag starts with 'B', the tag of the first subtoken should also start with B,
    if ner_tag[0] == 'B':    #  while the succedding subtokens' tag all start with I
        in_tag = 'I' + ner_tag[1:]    

        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    else:    # otherwise, the full word's tag is copied across all subtokens
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    return ner_tags_rep, mask_


def extract_spans(tags:List[str]) -> Dict[tuple, str]:
    """
    Convert a list of tags in the form 
        ['O', 'O', 'B-CONST_DIR', 'O', 'B-VAR', 'I-VAR', 'I-VAR', 'O']
    to a dict of spans in the form (span: tag)
        {
            (0, 0):'O', 
            (1, 1):'O', 
            (2, 2):'CONST_DIR', 
            (3, 3):'O', 
            (4, 6):'VAR', 
            (7, 7):'O',
        }
    In particular:
        1. each 'O' tag takes up one key-val pair, and the span in the place of the key has length 1
        2. each entity tag (single-token or spans multiple tokens) takes up one key-val pair
        3. The spans are specified by the start and end indices (including ends on both sides)
    """
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        indicator = nt[0]    # indicator can be B, I, O, extracted from the first char of the tag
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]    # cur_tag is the raw tag without the indicator
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans


def _is_divider(line: str) -> bool:
    """
    Determine whether a line is a divider (contains no useful info).
    True if line is empty or is the start of a document. False otherwise.
    Example input that return True: '-DOCSTART-\t_\t_\tO\n'. 
    """
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":  # or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False


def get_tags(tokens, tags, tokenizer=None, start_token_pattern='▁'):
    tag_results = [], []
    index = 0
    tokens = tokenizer.convert_ids_to_tokens(tokens)
    for token, tag in zip(tokens, tags):
        if token == tokenizer.pad_token:
            continue

        if index == 0:
            tag_results.append(tag)

        elif token.startswith(start_token_pattern) and token != '▁́':
            tag_results.append(tag)
        index += 1

    return tag_results
