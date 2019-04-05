import random
from copy import deepcopy as c
from typing import Dict, Any, List, Tuple

import attr
from boltons.iterutils import pairwise
from cached_property import cached_property

from utils import compute_idx_spans, flatten


class Document:
    def __init__(self, raw_text, tokens, corefs, speakers, genre, filename):
        self.raw_text = raw_text
        self.tokens = tokens
        self.corefs = corefs
        self.speakers = speakers
        self.genre = genre
        self.filename = filename

        # Filled in at evaluation time.
        self.tags = None

    def __getitem__(self, idx):
        return (self.tokens[idx], self.corefs[idx], \
                self.speakers[idx], self.genre)

    def __repr__(self):
        return 'Document containing %d tokens' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    @property
    def clusters(self)->Dict[Any,List[Tuple[int,int,str]]]:
        return {i:[(c['start'],c['end'],' '.join(self.tokens[c['start']:c['end']+1])) for c in cluster] for i,cluster in enumerate(self.corefs)}

    @cached_property
    def sents(self):
        """ Regroup raw_text into sentences """

        # Get sentence boundaries
        sent_idx = [idx+1
                    for idx, token in enumerate(self.tokens)
                    if token in ['.', '?', '!']] # TODO(tilo): WTF!!!

        # Regroup (returns list of lists)
        return [self.tokens[i1:i2] for i1, i2 in pairwise([0] + sent_idx)]

    def spans(self):
        """ Create Span object for each span """
        return [Span(i1=i[0], i2=i[-1], id=idx,
                    speaker=self.speaker(i), genre=self.genre)
                for idx, i in enumerate(compute_idx_spans(self.sents))]

    def truncate(self, MAX=50):
        """ Randomly truncate the document to up to MAX sentences """
        if len(self.sents) > MAX:
            i = random.sample(range(MAX, len(self.sents)), 1)[0]
            tokens = flatten(self.sents[i-MAX:i])
            return self.__class__(c(self.raw_text), tokens,
                                  c(self.corefs), c(self.speakers),
                                  c(self.genre), c(self.filename))
        return self

    def speaker(self, i):
        """ Compute speaker of a span """
        if self.speakers[i[0]] == self.speakers[i[-1]]:
            return self.speakers[i[0]]
        return None


@attr.s(frozen=True, repr=False)
class Span:

    # Left / right token indexes
    i1 = attr.ib()
    i2 = attr.ib()

    # Id within total spans (for indexing into a batch computation)
    id = attr.ib()

    # Speaker
    speaker = attr.ib()

    # Genre
    genre = attr.ib()

    # Unary mention score, as tensor
    si = attr.ib(default=None)

    # List of candidate antecedent spans
    yi = attr.ib(default=None)

    # Corresponding span ids to each yi
    yi_idx = attr.ib(default=None)

    def __len__(self):
        return self.i2-self.i1+1

    def __repr__(self):
        return 'Span representing %d tokens' % (self.__len__())