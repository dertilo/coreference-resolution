import attr
import torch
from torch import nn as nn
from torch.nn import functional as F

from data_schema import Span, Document
from scoring import FFNN, DistanceEmbedder
from utils import compute_idx_spans, pad_and_stack, remove_overlapping


class MentionScorer(nn.Module):

    def __init__(self, embeds_dim, attn_dim, distance_dim):
        super().__init__()

        self.span_repr_dim = attn_dim*2 + embeds_dim + distance_dim
        self.attention = FFNN(attn_dim)#TODO(tilo): how is this an attention?
        self.width = DistanceEmbedder(distance_dim)
        self.ffnn = FFNN(self.span_repr_dim)

    def forward(self, context_enc, embeds, doc:Document, num_max_antecedents=250):

        spans = [Span(span_idxs[0], span_idxs[-1], id=i,
                    speaker=doc.speaker(span_idxs), genre=doc.genre)
                for i, span_idxs in enumerate(compute_idx_spans(doc.sents))]

        attns = self.attention(context_enc)

        # Regroup attn values, embeds into span representations
        # TODO: figure out a way to batch
        span_attns = [attns[span.start:span.end + 1] for span in spans]
        span_embeds = [embeds[span.start:span.end + 1] for span in spans]


        padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        padded_embeds, _ = pad_and_stack(span_embeds)

        attn_weights = F.softmax(padded_attns, dim=1)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        widths = self.width([len(s) for s in spans])

        # TODO: figure out a way to batch
        lstm_embed_start_end = torch.stack([torch.cat((context_enc[s.start], context_enc[s.end])) for s in spans])

        span_representations = torch.cat((lstm_embed_start_end, attn_embeds, widths), dim=1) # g_i in paper

        mention_scores = self.ffnn(span_representations)

        # Update span object attributes
        # (use detach so we don't get crazy gradients by splitting the tensors)
        spans = [
            attr.evolve(span, mention_score=mention_score)
            for span, mention_score in zip(spans, mention_scores.detach())
        ]

        spans = prune(spans, len(doc.tokens))

        def candidate_antecedent_spans(idx):
            return spans[max(0, idx-num_max_antecedents):idx]

        spans = [
            attr.evolve(span, antecedent_spans=candidate_antecedent_spans(idx))
            for idx, span in enumerate(spans)
        ]

        return spans, span_representations, mention_scores


def prune(spans, T, LAMBDA=0.40):
    """ Prune mention scores to the top lambda percent.
    Returns list of tuple(scores, indices, g_i) """

    # Only take top λT spans, where T = len(doc)
    STOP = int(LAMBDA * T)

    # Sort by mention score, remove overlapping spans, prune to top λT spans
    sorted_spans = sorted(spans, key=lambda s: s.mention_score, reverse=True)
    nonoverlapping = remove_overlapping(sorted_spans)
    pruned_spans = nonoverlapping[:STOP]

    # Resort by start, end indexes
    spans = sorted(pruned_spans, key=lambda s: (s.start, s.end))

    return spans