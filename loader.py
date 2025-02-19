import gzip
import json
from typing import List
import numpy as np
import torch
from torchtext.vocab import Vectors

import os, io, re
from fnmatch import fnmatch
from cached_property import cached_property

from data_schema import Document
from utils import to_cuda

NORMALIZE_DICT = {"/.": ".", "/?": "?",
                  "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}",
                  "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["/", "%", "*"]


class Corpus:
    def __init__(self, documents):
        self.docs = documents
        self.vocab, self.char_vocab = self.get_vocab()

    def __getitem__(self, idx):
        return self.docs[idx]

    def __repr__(self):
        return 'Corpus containg %d documents' % len(self.docs)

    def get_vocab(self):
        """ Set vocabulary for LazyVectors """
        vocab, char_vocab = set(), set()

        for document in self.docs:
            vocab.update(document.tokens)
            char_vocab.update([char
                               for word in document.tokens
                               for char in word])

        return vocab, char_vocab


class LazyVectors:
    """Load only those vectors from GloVE that are in the vocab.
    Assumes PAD id of 0 and UNK id of 1
    """

    unk_idx = 1

    def __init__(self, name,
                       cache,
                       skim=None,
                       vocab=None):
        """  Requires the glove vectors to be in a folder named .vector_cache
        Setup:
            >> cd ~/where_you_want_to_save
            >> mkdir .vector_cache
            >> mv ~/where_glove_vectors_are_stored/glove.840B.300d.txt
                ~/where_you_want_to_save/.vector_cache/glove.840B.300d.txt
        Initialization (first init will be slow):
            >> VECTORS = LazyVectors(cache='~/where_you_saved_to/.vector_cache/',
                                     vocab_file='../path/vocabulary.txt',
                                     skim=None)
        Usage:
            >> weights = VECTORS.weights()
            >> embeddings = torch.nn.Embedding(weights.shape[0],
                                              weights.shape[1],
                                              padding_idx=0)
            >> embeddings.weight.data.copy_(weights)
            >> embeddings(sent_to_tensor('kids love unknown_word food'))
        You can access these moved vectors from any repository
        """
        self.__dict__.update(locals())
        if self.vocab is not None:
            self.set_vocab(vocab)

    @classmethod
    def from_corpus(cls, corpus_vocabulary, name, cache):
        return cls(name=name, cache=cache, vocab=corpus_vocabulary)

    @cached_property
    def loader(self):
        return Vectors(self.name, cache=self.cache)

    def set_vocab(self, vocab):
        """ Set corpus vocab
        """
        # Intersects and initializes the torchtext Vectors class
        self.vocab = [v for v in vocab if v in self.loader.stoi][:self.skim]

        self.set_dicts()

    def get_vocab(self, filename):
        """ Read in vocabulary (top 30K words, covers ~93.5% of all tokens) """
        return read_file(filename) #TODO(tilo): the-FAQ!

    def set_dicts(self):
        """ _stoi: map string > index
            _itos: map index > string
        """
        self._stoi = {s: i for i, s in enumerate(self.vocab)}
        self._itos = {i: s for s, i in self._stoi.items()}

    def weights(self):
        """Build weights tensor for embedding layer """
        # Select vectors for vocab words.
        weights = torch.stack([
            self.loader.vectors[self.loader.stoi[s]]
            for s in self.vocab
        ])

        # Padding + UNK zeros rows.
        return torch.cat([
            torch.zeros((2, self.loader.dim)),
            weights,
        ])

    def stoi(self, s):
        """ String to index (s to i) for embedding lookup """
        idx = self._stoi.get(s)
        return idx + 2 if idx else self.unk_idx

    def itos(self, i):
        """ Index to string (i to s) for embedding lookup """
        token = self._itos.get(i)
        return token if token else 'UNK'


def read_corpus(dirname):
    # conll_files = parse_filenames(dirname=dirname, pattern="*gold_conll")
    return Corpus(load_file_scisci_format(dirname))

def read_jsons_from_file(file, limit=np.Inf):
    with gzip.open(file, mode="rb") if file.endswith('.gz') else open(file, mode="rb") as f:
        counter=0
        for line in f:
            # assert isinstance(line,bytes)
            counter += 1
            if counter > limit: break
            yield json.loads(line.decode('utf-8'))

def load_file_scisci_format(filename)->List[Document]:
    def process_json(d):
        coref_clusters = [[
            {'label': i,
             'start': start,
             'end': end,
             'span': (start, end)
             }
            for start, end in cluster]
            for i, cluster in enumerate(d['clusters'])]
        tokens = [token for sentence in d['sentences'] for token in sentence]
        raw_text = ' '.join(tokens)

        coref_clusters = [c for cluster in coref_clusters for c in cluster]
        doc = Document(raw_text, tokens, coref_clusters, 'deineMutter' * len(tokens), 'boringShit', filename)
        return doc

    return [process_json(json) for json in read_jsons_from_file(filename)]

def parse_filenames(dirname, pattern = "*conll"):
    """ Walk a nested directory to get all filename ending in a pattern """
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, pattern):
                yield os.path.join(path, name)

def clean_token(token):
    """ Substitute in /?(){}[] for equivalent CoNLL-2012 representations,
    remove /%* """
    cleaned_token = token
    if cleaned_token in NORMALIZE_DICT:
        cleaned_token = NORMALIZE_DICT[cleaned_token]

    if cleaned_token not in REMOVED_CHAR:
        for char in REMOVED_CHAR:
            cleaned_token = cleaned_token.replace(char, u'')

    if len(cleaned_token) == 0:
        cleaned_token = ","
    return cleaned_token

def lookup_tensor(tokens, vectorizer):
    """ Convert a sentence to an embedding lookup tensor """
    return to_cuda(torch.tensor([vectorizer.stoi(t) for t in tokens]))

# if __name__ == '__main__':
    # Load in corpus, lazily load in word vectors.
train_corpus = read_corpus('/home/tilo/code/NLP/IE/sciie/data/processed_data/json/train.json')
val_corpus = read_corpus('/home/tilo/code/NLP/IE/sciie/data/processed_data/json/train.json')
test_corpus = read_corpus('/home/tilo/code/NLP/IE/sciie/data/processed_data/json/test.json')
    #
GLOVE = LazyVectors.from_corpus(train_corpus.vocab,
                                name='glove.840B.300d.txt',
                                cache='/home/tilo/data/embeddings')

# TURIAN = LazyVectors.from_corpus(train_corpus.vocab,
#                                  name='hlbl-embeddings-scaled.EMBEDDING_SIZE=50',
#                                  cache='/Users/sob/github/.vector_cache/')
