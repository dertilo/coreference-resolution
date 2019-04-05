from typing import List

import torch
from torch import nn as nn
from torch.nn import functional as F

from loader import GLOVE, lookup_tensor, train_corpus
from utils import pack, unpack_and_unpad, to_cuda


class DocumentEncoder(nn.Module):
    """ Document encoder for tokens
    """
    def __init__(self, hidden_dim, char_filters, n_layers=2):
        super().__init__()

        # Unit vector embeddings as per Section 7.1 of paper
        glove_weights = F.normalize(GLOVE.weights())
        # turian_weights = F.normalize(TURIAN.weights())

        # GLoVE
        self.glove = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
        self.glove.weight.data.copy_(glove_weights)
        self.glove.weight.requires_grad = False

        # Turian
        # self.turian = nn.Embedding(turian_weights.shape[0], turian_weights.shape[1])
        # self.turian.weight.data.copy_(turian_weights)
        # self.turian.weight.requires_grad = False

        # Character
        self.char_embeddings = CharCNN(char_filters)

        # Sentence-LSTM
        self.lstm = nn.LSTM(glove_weights.shape[1]+char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout
        self.emb_dropout = nn.Dropout(0.50, inplace=True)
        self.lstm_dropout = nn.Dropout(0.20, inplace=True)

    def forward(self, sents:List[List[str]]):
        embeds = [self.embed(s) for s in sents]

        packed, reorder = pack(embeds)
        self.emb_dropout(packed[0])

        output = self.contextual_encoding(packed)

        contextual_encoded = unpack_and_unpad(output, reorder)

        return torch.cat(contextual_encoded, dim=0), torch.cat(embeds, dim=0)

    def contextual_encoding(self, packed):
        output, _ = self.lstm(packed)
        self.lstm_dropout(output[0])
        return output

    def embed(self, sent):
        glove_embeds = self.glove(lookup_tensor(sent, GLOVE))
        # tur_embeds = self.turian(lookup_tensor(sent, TURIAN))
        char_embeds = self.char_embeddings(sent)

        return torch.cat((glove_embeds, char_embeds), dim=1)


class CharCNN(nn.Module):
    """ Character-level CNN. Contains character embeddings.
    """

    unk_idx = 1
    vocab = train_corpus.char_vocab
    _stoi = {char: idx+2 for idx, char in enumerate(vocab)}
    pad_size = 15

    def __init__(self, filters, char_dim=8):
        super().__init__()

        self.embeddings = nn.Embedding(len(self.vocab)+2, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.pad_size,
                                              out_channels=filters,
                                              kernel_size=n) for n in (3,4,5)])

    def forward(self, sent):
        """ Compute filter-dimensional character-level features for each doc token """
        embedded = self.embeddings(self.sent_to_tensor(sent))
        convolved = torch.cat([F.relu(conv(embedded)) for conv in self.convs], dim=2)
        pooled = F.max_pool1d(convolved, convolved.shape[2]).squeeze(2)
        return pooled

    def sent_to_tensor(self, sent):
        """ Batch-ify a document class instance for CharCNN embeddings """
        tokens = [self.token_to_idx(t) for t in sent]
        batch = self.char_pad_and_stack(tokens)
        return batch

    def token_to_idx(self, token):
        """ Convert a token to its character lookup ids """
        return to_cuda(torch.tensor([self.stoi(c) for c in token]))

    def char_pad_and_stack(self, tokens):
        """ Pad and stack an uneven tensor of token lookup ids """
        skimmed = [t[:self.pad_size] for t in tokens]

        lens = [len(t) for t in skimmed]

        padded = [F.pad(t, (0, self.pad_size-length))
                  for t, length in zip(skimmed, lens)]

        return torch.stack(padded)

    def stoi(self, char):
        """ Lookup char id. <PAD> is 0, <UNK> is 1. """
        idx = self._stoi.get(char)
        return idx if idx else self.unk_idx