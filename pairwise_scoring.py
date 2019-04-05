from typing import List

import attr
import torch
from torch import nn as nn
from torch.nn import functional as F

from data_schema import Span
from scoring import DistanceEmbedder, FFNN
from utils import speaker_label, to_cuda, pairwise_indexes, to_var, pad_and_stack


class PairwiseScorer(nn.Module):

    def __init__(self, span_repr_dim, distance_dim, genre_dim, speaker_dim):
        super().__init__()

        phi_feature_dim = distance_dim + genre_dim + speaker_dim
        ant_scorer_input_dim = span_repr_dim * 3 + phi_feature_dim

        self.distance = DistanceEmbedder(distance_dim)
        self.genre = Genre(genre_dim)
        self.speaker = Speaker(speaker_dim)

        self.score = FFNN(ant_scorer_input_dim)

    def forward(self, spans:List[Span], span_representations, mention_scores):

        mention_ids, antecedent_ids, distances, genres, speakers = zip(*[
            (span.id, ant_span.id, span.end-ant_span.start, span.genre, speaker_label(span, ant_span))
                                                for span in spans
                                                for ant_span in span.antecedent_spans])

        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))

        phi_feature = torch.cat((self.distance(distances),
                         self.genre(genres),
                         self.speaker(speakers)), dim=1)

        def get_representations(ids):
            return torch.index_select(span_representations,0,ids)

        m_r = get_representations(mention_ids)
        a_r = get_representations(antecedent_ids)

        pairs = torch.cat((m_r, a_r, m_r*a_r, phi_feature), dim=1)


        def get_mention_scores(ids):
            return torch.index_select(mention_scores,0,ids)

        m_s = get_mention_scores(mention_ids)#in paper: s_i
        a_s = get_mention_scores(antecedent_ids)#in paper: s_j
        am_s = self.score(pairs)#in paper: s_ij

        coref_scores = torch.sum(torch.cat((m_s, a_s, am_s), dim=1), dim=1, keepdim=True)

        spans = [
            attr.evolve(span,
                        antspan_span=[((ant_span.start, ant_span.end), (span.start, span.end)) for ant_span in span.antecedent_spans]
                        )
            for span, score in zip(spans, coref_scores)
        ]

        list_num_ant_spans = [len(s.antecedent_spans) for s in spans if len(s.antecedent_spans)>0]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [to_cuda(torch.tensor([]))] \
                         + list(torch.split(coref_scores, list_num_ant_spans, dim=0))

        epsilon = to_var(torch.tensor([[0.]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        # Batch and softmax
        # get the softmax of the scores for each span in the document given
        probs = [F.softmax(tensr,dim=0) for tensr in with_epsilon]

        # pad the scores for each one with a dummy value, 1000 so that the tensors can
        # be of the same dimension for calculation loss and what not.
        padded_probs, _ = pad_and_stack(probs, value=1000).squeeze()
        return spans, padded_probs


class Genre(nn.Module):
    """ Learned continuous representations for genre. Zeros if genre unknown.
    """

    genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    _stoi = {genre: idx+1 for idx, genre in enumerate(genres)}

    def __init__(self, genre_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(len(self.genres)+1, genre_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, labels):
        """ Embedding table lookup """
        return self.embeds(self.stoi(labels))

    def stoi(self, labels):
        """ Locate embedding id for genre """
        indexes = [self._stoi.get(gen) for gen in labels]
        return to_cuda(torch.tensor([i if i is not None else 0 for i in indexes]))


class Speaker(nn.Module):
    """ Learned continuous representations for binary speaker. Zeros if speaker unknown.
    """

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(3, speaker_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, speaker_labels):
        """ Embedding table lookup (see src.utils.speaker_label fnc) """
        return self.embeds(to_cuda(torch.tensor(speaker_labels)))