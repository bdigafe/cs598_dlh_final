import random


def age_vocab(max_age, mon=1, symbol=None):
    """
        Parameters
            max_age: int = the maximum age of the dataset

            mon: [1, 12] = The granularity of the age. 
                1 = Monthly (12 indexes per year)
                12=Yearly (1 index per year)

        Retrurns two dictionaries that map age to index and index to age
    """
    age2idx = {}
    idx2age = {}

    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])

    if mon == 12:
        for i in range(max_age):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    elif mon == 1:
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    else:
        age2idx = None
        idx2age = None
    return age2idx, idx2age


def code2index(tokens, token2idx, mask_token=None):
    """"
        parameters:
            tokens: list = list of tokens
            token2idx: dict = dictionary that maps tokens to indexes
            mask_token: str = value of unknown tokens

        returns:
            tokens: list = list of tokens (same as input)
            output_tokens: list = list of indexes

        example:
            tokens: ['UNK', 'D1', 'D2', 'D3',  'D5']
            token2idx: {'UNK' : 0,'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4, 'D5': 5}
            mask_token: 'UNK'

            output_tokens: [0, 1, 2, 3, 5]
    """
    output_tokens = []

    for _, token in enumerate(tokens):
        if token == mask_token:
            output_tokens.append(token2idx['UNK'])
        else:
            output_tokens.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_tokens


def random_mask(tokens, token2idx):
    """
        Mask some random tokens for masked language modeling task with probabilities as in the original BERT paper.

        parameters:
            tokens: list = list of tokens
            token2idx: dict = dictionary that maps tokens to indexes

        returns:
            tokens: list = list of tokens (same as input)

            tokens_masked: list = list of masked tokens
                15% of the time the original token will be masked:
                    80% the token is replaced with [MASK]
                    10% the token is replaced with a random token
                    rest: the same token is retained

            tokens_label: list = list of labels (original value) for masked tokens
    """

    output_label = []
    output_token = []

    for i, token in enumerate(tokens):
        prob = random.random()

        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token (noise)
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def index_seg(tokens, symbol='SEP'):
    """
        The segment index is used to differentiate two consequent visits.
        The value of the segment is 0 or 1 alternating for each sentence: [0, 1, 0, 1, 0, 1]
        Tokens are separated by the symbol will be assigned the same segment index.

        Example: 
            Data:   [v1[D1], v2[D1,D2], v3[D1,D2,D3]]

                    ----V1----  ----V2------  -----V3--------
            tokens: [D1,        SEP, D1, D2,  SEP, D1, D2, D3]

                    ----V1----  ----V2------  -----V3--------
            seg:    [0,   0,    1,   1,  1,   0,   0,  0,  0]

        Parameters:
            tokens: list = list of tokens
            symbol: str = symbol to segment the tokens

        returns: seg = list of segment indexes
    """
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)

    return seg


def position_idx(tokens, symbol='SEP'):
    """
        The position index is used to differentiate two consequent visits.
        The value of the position is the index of the token in the sentence: [0, 1, 2, 3, 4, 5]
        Tokens are separated by the symbol will be assigned the same position index.

        Example: 
        Data:   [v1[D1], v2[D1,D2], v3[D1,D2,D3]]

                ---V1----  -----V2------  -------V3---------
        tokens: [ D1,       SEP, D1, D2,  SEP, D1, D2, D3]

                ----V1----  ----V2------  -----V3--------
        pos:    [0,   0,    1,   1,  1,   2,   2,  2,  2]

    """
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def seq_padding(tokens, max_len, token2idx=None, symbol=None, unkown=True):
    """
        Add padding to the tokens to make the length of the tokens equal to max_len.
        parameters:
            tokens: list = list of tokens
            max_len: int = max length of the tokens
            token2idx: dict = dictionary that maps tokens to indexes
            symbol: str = symbol to pad the tokens
            unkown: bool = if True, the tokens that are not in the token2idx will be replaced by UNK

        returns: seq = list of tokens with padding

        Example 1:
            tokens: ['I', 'am', 'a', 'student']
            max_len: 6
            token2idx: {'I': 1, 'am': 2, 'a': 3, 'student': 4}
            symbol: 'PAD'
            unkown: True, index of UNK is 0
            seq: [1, 2, 3, 4, 0, 0]

        Example 2:
            tokens: ['I', 'am', 'a', 'student']
            max_len: 6
            token2idx: None
            symbol: 'PAD'
            unkown: True
            seq: ['I', 'am', 'a', 'student', 'PAD', 'PAD']

    """
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)

    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                if unkown:
                    seq.append(token2idx.get(tokens[i], token2idx['UNK']))
                else:
                    seq.append(token2idx.get(tokens[i]))
            else:
                seq.append(token2idx.get(symbol))
    return seq
