import numpy as np
from collections import Counter
import itertools

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import constants as constant


np.random.seed(13)


def Y_to_y(Y):
    y = np.argmax(Y, axis=2)
    y = np.reshape(y, (y.shape[0] * y.shape[1]))
    return y


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, max_sent_length, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_sent_length)

    return np.array(sequence_padded), sequence_length


def merge_dataset(d1, d2):
    """
    merge 2 datasets into only one dataset
    :param Dataset d1:
    :param Dataset d2:
    :return Dataset:
    """
    r = Dataset(data_name='merged')

    r.words = list(itertools.chain(d1.words, d2.words))
    r.labels = list(itertools.chain(d1.labels, d2.labels))
    r.poses = list(itertools.chain(d1.poses, d2.poses))
    r.relations = list(itertools.chain(d1.relations, d2.relations))
    r.directions = list(itertools.chain(d1.directions, d2.directions))
    r.identities = list(itertools.chain(d1.identities, d2.identities))
    r.wordnet_supersets = list(itertools.chain(d1.wordnet_supersets, d2.wordnet_supersets))

    return r


class Dataset:
    def __init__(self, data_name, vocab_words=None, vocab_poses=None, vocab_depends=None):
        self.data_name = data_name

        self.words = None
        self.labels = None
        self.poses = None
        self.relations = None
        self.directions = None
        self.identities = None

        self.vocab_words = vocab_words
        self.vocab_poses = vocab_poses
        self.vocab_depends = vocab_depends

        self._process_data()
        self._clean_data()

    def _clean_data(self):
        del self.vocab_words
        del self.vocab_poses
        del self.vocab_depends

    def _process_data(self):
        with open(self.data_name, 'r') as f:
            raw_data = f.readlines()
        data_words, data_y, data_lens, data_pos, data_relations, data_directions, self.identities = self.parse_raw(raw_data)
        words = []
        labels = []
        poses = []
        relations = []
        directions = []
        for i in range(len(data_words)):
            rs = []
            for r in data_relations[i]:
                rid = self.vocab_depends[r]
                rs += [rid]
            relations.append(rs)

            ds = []
            for d in data_directions[i]:
                did = 1 if d == 'l' else 2
                ds += [did]
            directions.append(ds)

            ws, ps, wns = [], [], []
            for w, p in zip(data_words[i], data_pos[i]):
                if w in self.vocab_words:
                    word_id = self.vocab_words[w]
                else:
                    word_id = self.vocab_words[constant.UNK]
                ws.append(word_id)

                p_id = self.vocab_poses[p]
                ps += [p_id]
            words.append(ws)
            poses.append(ps)

            lb = constant.ALL_LABELS.index(data_y[i][0])
            labels.append(lb)

        self.words = words
        self.labels = labels
        self.poses = poses
        self.relations = relations
        self.directions = directions

    def parse_raw(self, raw_data):
        all_words = []
        all_relations = []
        all_directions = []
        all_poses = []
        all_labels = []
        all_lens = []
        all_identities = []
        pmid = ''
        for line in raw_data:
            l = line.strip().split()
            if len(l) == 1:
                pmid = l[0]
            else:
                try:
                    pair = l[0]
                    label = l[1]
                    if label:
                        joint_sdp = ' '.join(l[2:])
                        sdps = joint_sdp.split("-PUNC-")
                        for sdp in sdps:
                            # S xuoi
                            nodes = sdp.split()
                            words = []
                            poses = []
                            relations = []
                            directions = []

                            for node in nodes:
                                node = node.rsplit('/', 1)
                                if len(node) == 2:
                                    word = constant.UNK if node[0] == '' else node[0]
                                    pos = 'NN' if node[-1] == '' else node[-1]

                                    words.append(word)
                                    poses.append(pos)
                                else:
                                    dependency = node[0]
                                    r = '(' + dependency[3:]
                                    d = dependency[1]
                                    r = r.split(':', 1)[0] + ')' if ':' in r else r
                                    relations.append(r)
                                    directions.append(d)

                            all_words.append(words)
                            all_lens.append(len(words))
                            all_relations.append(relations)
                            all_directions.append(directions)
                            all_poses.append(poses)
                            all_labels.append([label])
                            all_identities.append((pmid, pair))
                    else:
                        print(l)
                except Exception as e:
                    print(e.__str__())

        return all_words, all_labels, all_lens, all_poses, all_relations, all_directions, all_identities


    @staticmethod
    def reverse_sdp(sdp):
        if sdp:
            nodes = sdp.split()
            if len(nodes) % 2:
                ret = []
                for i, node in enumerate(nodes[::-1]):
                    if i % 2:
                        rev_dep = '({}_{})'.format(
                            'r' if node[1] == 'l' else 'l',
                            node[3:-1]
                        )
                        ret.append(rev_dep)
                    else:
                        ret.append(node)

                return ' '.join(ret)
            else:
                raise ValueError('Invalid sdp')
        else:
            return ''

    def __apply_indicates(self, indicates):
        self.words = [self.words[i] for i in indicates]
        self.labels = [self.labels[i] for i in indicates]
        self.poses = [self.poses[i] for i in indicates]
        self.relations = [self.relations[i] for i in indicates]
        self.directions = [self.directions[i] for i in indicates]
        self.identities = [self.identities[i] for i in indicates]
        self.wordnet_supersets = [self.wordnet_supersets[i] for i in indicates]

    def under_sample(self, n, seed):
        c = Counter(self.labels)
        print('training shape before under sampling: {}'.format({k: c[k] for k in c}))

        d = {k: n for k in c if c[k] > n}
        rus = RandomUnderSampler(ratio=d, random_state=seed, return_indices=True)

        sample_data = [[0]] * len(self.labels)
        _, _, indicates = rus.fit_sample(sample_data, self.labels)
        self.__apply_indicates(indicates)

        c = Counter(self.labels)
        print('training shape after under sampling: {}'.format({k: c[k] for k in c}))

    def over_sample(self, n, seed):
        c = Counter(self.labels)
        print('training shape before over sampling: {}'.format({k: c[k] for k in c}))

        d = {k: n for k in c if c[k] < n}
        ros = RandomOverSampler(ratio=d, random_state=seed)

        sample_data = [[i] for i in range(len(self.labels))]
        data, _ = ros.fit_sample(sample_data, self.labels)
        indicates = [i[0] for i in data]
        self.__apply_indicates(indicates)

        c = Counter(self.labels)
        print('training shape before over sampling: {}'.format({k: c[k] for k in c}))
