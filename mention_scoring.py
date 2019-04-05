import attr
import torch
from torch import nn as nn
from torch.nn import functional as F

from loader import Span
from scoring import Score, Distance
from utils import compute_idx_spans, pad_and_stack, remove_overlapping


class MentionScorer(nn.Module):
    """ Mention scoring module
    """
    def __init__(self, gi_dim, attn_dim, distance_dim):
        super().__init__()

        self.attention = Score(attn_dim)#TODO(tilo): how is this an attention?
        self.width = Distance(distance_dim)
        self.score = Score(gi_dim)

    def forward(self, states, embeds, doc, K=250):
        """ Compute unary mention score for each span
        """

        # Initialize Span objects containing start index, end index, genre, speaker
        spans = [Span(i1=i[0], i2=i[-1], id=idx,
                      speaker=doc.speaker(i), genre=doc.genre)
                 for idx, i in enumerate(compute_idx_spans(doc.sents))]

        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(states)

        # Regroup attn values, embeds into span representations
        # TODO: figure out a way to batch
        span_attns, span_embeds = zip(*[(attns[s.i1:s.i2+1], embeds[s.i1:s.i2+1]) for s in spans])

        # Pad and stack span attention values, span embeddings for batching
        padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        padded_embeds, _ = pad_and_stack(span_embeds)

        # Weight attention values using softmax
        attn_weights = F.softmax(padded_attns, dim=1)

        # Compute self-attention over embeddings (x_hat)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        # Compute span widths (i.e. lengths), embed them
        widths = self.width([len(s) for s in spans])

        # Get LSTM state for start, end indexes
        # TODO: figure out a way to batch
        start_end = torch.stack([torch.cat((states[s.i1], states[s.i2]))
                                 for s in spans])

        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((start_end, attn_embeds, widths), dim=1)

        # Compute each span's unary mention score
        mention_scores = self.score(g_i)

        # Update span object attributes
        # (use detach so we don't get crazy gradients by splitting the tensors)
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        # Prune down to LAMBDA*len(doc) spans
        spans = prune(spans, len(doc))

        # Update antencedent set (yi) for each mention up to K previous antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores


def prune(spans, T, LAMBDA=0.40):
    """ Prune mention scores to the top lambda percent.
    Returns list of tuple(scores, indices, g_i) """

    # Only take top λT spans, where T = len(doc)
    STOP = int(LAMBDA * T)

    # Sort by mention score, remove overlapping spans, prune to top λT spans
    sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True)
    nonoverlapping = remove_overlapping(sorted_spans)
    pruned_spans = nonoverlapping[:STOP]

    # Resort by start, end indexes
    spans = sorted(pruned_spans, key=lambda s: (s.i1, s.i2))

    return spans