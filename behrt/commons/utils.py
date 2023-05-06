
import os
import pickle
import random
import numpy as np
from pandas import DataFrame


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(data_path, sample_size: int = 0):
    data = pickle.load(open(data_path, 'rb'))

    if (sample_size > 0):
        data = data[:sample_size]

    return data


def get_codes_vocab(vocab_path):
    """
        purpose: Load the BERTconditions codes vocab from the pickle file.
        parameters: None
        returns: bertVocab: dict = dictionary that maps tokens to indexes and indexes to tokens
    """
    condition_codes = pickle.load(open(vocab_path, 'rb'))

    bertToken2Index = {
        'CLS': 0,
        'SEP': 1,
        'PAD': 2,
        'MASK': 3,
        'UNK': 4,
    }

    idx = 5
    for cond in condition_codes["condition"]:
        if (cond not in ('CLS', 'SEP', 'PAD', 'MASK', 'UNK')):
            bertToken2Index[cond] = idx
            idx += 1

    bertIndex2Token = dict()

    for x in bertToken2Index:
        bertIndex2Token[bertToken2Index[x]] = x

    bertVocab = {
        'token2idx': bertToken2Index,
        'idx2token': bertIndex2Token
    }

    return bertVocab


def age_vocab(max_age, mon=1, symbol=None):
    """
        Parameters
            max_age: int = the maximum age of the dataset
            mon: [1, 12] = The granularity of the age. 
                1   = Monthly (12 indexes per year)
                12  = Yearly (1 index per year)

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


def code2index(seq_tokens, token2idx, mask_token=None):
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

    for _, token in enumerate(seq_tokens):
        if token == mask_token:
            output_tokens.append(token2idx['UNK'])
        else:
            output_tokens.append(token2idx.get(token, token2idx['UNK']))

    return seq_tokens, output_tokens


def random_mask(seq_tokens, token2idx):
    """
        Mask some random tokens for masked language modeling task with probabilities as in the original BERT paper.

        parameters:
            tokens      : list = list of tokens
            token2idx   : dict = dictionary that maps tokens to indexes

        returns:
            tokens_masked: list = list of masked tokens
                15% of the time the original token will be masked:
                    80% the token is replaced with [MASK]
                    10% the token is replaced with a random token
                    rest: the same token is retained

            tokens_label: list = list of labels (original value) for masked tokens
    """

    output_label = []
    output_token = []

    for _, token in enumerate(seq_tokens):
        prob = random.random()

        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token (noise)
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token
            else:
                output_token.append(token2idx.get(token, token2idx['UNK']))

            output_label.append(token2idx.get(token, token2idx['UNK']))

        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return output_token, output_label


def index_seg(seq_tokens, symbol='SEP'):
    """
        The segment index is used to differentiate two consecutive visits.
        The value of the segment is 0 or 1 alternating for each sentence: [0, 1, 0, 1, 0, 1]
        Tokens are separated by the symbol will be assigned the same segment index.

        Example: 
            Data:   [v1[D1], v2[D1,D2], v3[D1,D2,D3]]

                    ----V1----  ----V2------  -----V3------------
            tokens: [D1,        SEP, D1, D2,  SEP, D1, D2, D3 SEP]

                    ----V1----  ----V2------  -----V3-------------
            seg:    [0,   0,    1,   1,  1,   1,   0,  0,  0, 0]

        Parameters:
            tokens: list = list of tokens
            symbol: str = symbol to segment the tokens

        returns: seg = list of segment indexes
    """
    flag = 0
    seg = []

    for token in seq_tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)

    return seg


def position_idx(seq_tokens, symbol='SEP'):
    """
        The position index is used to encode the positional of a diagnosis code in the entire EHR of the patient.
        The value of the position is the index of the token in the sentence: [0, 1, 2, 3, 4, 5]
        Condition codes in the same visit (between two SEP tokens) will have the same position index.

        Example: 
        Data:   [v1[D1], v2[D1,D2], v3[D1,D2,D3]]

                ---V1----   -----V2------  -------V3---------
        tokens: [ D1, SEP,  D1, D2, SEP,   D1, D2, D3, SEP]

                ----V1----  ----V2------   -----V3-----------
        pos:    [0, 0,      1,  1, 1,      2,  2,  2,  2]

    """
    pos = []
    flag = 0

    for token in seq_tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def seq_padding(seq_tokens, max_len, token2idx=None, symbol=None, unknown=True):
    """
        Add padding to tokens to make the length of tokens equal to max_len.

        parameters:
            seq_tokens: list = list of tokens
            max_len: int = max length of the tokens
            token2idx: dict = dictionary that maps tokens to indexes
            symbol: str = symbol to pad the tokens
            unkown: bool = if True, the tokens that are not in the token2idx will be replaced by UNK

        returns: seq = list of tokens with padding

        Example 1:
            tokens: ['I', 'am', 'a', 'student']
            max_len: 6
            token2idx: {'I': 1, 'am': 2, 'a': 3, 'student': 4}
            symbol: 'PAD', index =0 
            unkown: False
            seq: [1, 2, 3, 4, 0, 0]
    """
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(seq_tokens)

    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(seq_tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                if unknown:
                    seq.append(token2idx.get(seq_tokens[i], token2idx['UNK']))
                else:
                    seq.append(token2idx.get(seq_tokens[i]))
            else:
                seq.append(token2idx.get(symbol))
    return seq


def remove_system_codes_from_token_dict(token2idx):
    token2idx = token2idx.copy()

    del token2idx['PAD']
    del token2idx['SEP']
    del token2idx['CLS']
    del token2idx['MASK']
    del token2idx['UNK']

    token = list(token2idx.keys())

    labelVocab = {}
    for i, x in enumerate(token):
        labelVocab[x] = i

    return labelVocab


def get_patient_visits(conditions, age_seqs):
    all_visits_conditions = []
    visit_conditions = []

    if len(conditions) != len(age_seqs):
        print(
            f'Error: conditions={len(conditions)} and age_seqs={len(age_seqs)} have different lengths')

    c = 0
    all_age_visits = []
    age_visits = []

    for code in conditions:
        if (code == 'SEP'):
            visit_conditions.append('SEP')
            all_visits_conditions.append(visit_conditions)
            visit_conditions = []

            age_visits.append(age_seqs[c])
            all_age_visits.append(age_visits)
            age_visits = []

        else:
            if (code not in visit_conditions):
                visit_conditions.append(code)
            age_visits.append(age_seqs[c])

        if (c < len(age_seqs)-1):
            c += 1

    return all_visits_conditions, all_age_visits


def get_multi_hot_vector(visit, code2idx):
    vector = np.zeros(len(code2idx))

    for code in visit:
        if (code in code2idx):
            vector[code2idx[code]] = 1

    return vector


def get_labelled_data(data, min_size: int, seq_ages, num_months: int = 0):

    all_pids = []
    all_visits = []
    all_ages = []
    all_labels = []

    for idx in range(len(data)):
        visits, ages = get_patient_visits(
            data.iloc[idx]['conditions'], seq_ages.iloc[idx]['ages'])

        if (len(visits) <= min_size + 1):
            continue

        j = 0
        if (num_months > 0):
            prev_age = int(ages[min_size-1][0])

            for i in range(min_size, len(ages)):
                if (int(ages[i][0]) - prev_age >= num_months):
                    j = i
                    break
            # if there is no visit after num_months, skip this patient
            if (j == 0):
                continue
        else:
            j = random.randint(min_size, len(visits)-2)

        all_pids.append(data.iloc[idx]["pid"])
        xp = visits[:j]
        xp = [visit for visits in xp for visit in visits]

        yp = visits[j]

        xp_ages = ages[:j]
        xp_ages = [age for ages in xp_ages for age in ages]

        all_ages.append(xp_ages)
        all_visits.append(xp)
        all_labels.append(yp)

    return DataFrame({'pid': all_pids, 'visit': all_visits, 'age': all_ages, 'label': all_labels})


def split_data(data, seq_ages, train_ratio=0.8, min_size=4, num_months=0):
    """
        Generate train and test dataset from the original dataset.
        The data dataset contains the EHR of all patients.
        The train dataset contains the EHR of the training patients.
        The test dataset contains the EHR of the testing patients.

        parameters:
            data: list = list of patients
            code2idx: dict = dictionary that maps codes to indexes
            train_ratio: float = ratio of the number of training patients to the number of testing patients
            min_size: int = minimum number of visits in a patient
    """

    # Split data into train and test

    train_size = int(len(data) * train_ratio)

    train_data = data[:train_size]
    test_data = data[train_size:]

    train_age_seqs = seq_ages[:train_size]
    test_age_seqs = seq_ages[train_size:]

    train_dataset = get_labelled_data(
        train_data, min_size, train_age_seqs, num_months)
    test_dataset = get_labelled_data(
        test_data, min_size, test_age_seqs, num_months)

    """  
    data = get_labelled_data(data, min_size, seq_ages, num_months=num_months)
    train_size = int(len(data) * train_ratio)

    train_dataset = data[train_size:]
    test_dataset = data[:train_size]
    """

    return train_dataset, test_dataset
