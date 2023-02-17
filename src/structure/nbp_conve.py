
import torch
from torch import nn
import math

from .neural_binary_predicate import NeuralBinaryPredicate


class ConvEScorer(nn.Module):
    r"""Implementation of the ConvE KGE scorer. by KGE class
    Must be used with ReciprocalRelationsModel."""
    def __init__(self, 
                 emb_dim,
                 aspect_ratio=2,
                 filter_size=3,
                 stride=1,
                 padding=0,
                 feature_map_dropout=0.2,
                 projection_dropout=0.3,
                 convolution_bias=True):

        super(ConvEScorer, self).__init__()
        # self.configuration_key = configuration_key
        self.emb_dim = emb_dim - 1 # keep one dimension for the bias
        aspect_ratio = aspect_ratio
        self.emb_height = math.sqrt(self.emb_dim / aspect_ratio)
        self.emb_width = self.emb_height * aspect_ratio

        # round embedding dimension to match aspect ratio
        if self.emb_dim % self.emb_height or self.emb_dim % self.emb_width:
            raise Exception(
                (
                    "Embedding dimension {} incompatible with aspect ratio {}; "
                    "width ({}) or height ({}) is not integer. "
                    "Adapt dimension or set conve.round_dim=true"
                ).format(self.emb_dim,
                         aspect_ratio,
                         self.emb_width,
                         self.emb_height)
            )

        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.feature_map_dropout = torch.nn.Dropout2d(
            feature_map_dropout
        )
        self.projection_dropout = torch.nn.Dropout(
            projection_dropout
        )

        self.convolution = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(self.filter_size, self.filter_size),
            stride=self.stride,
            padding=self.padding,
            bias=convolution_bias,
        )
        
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim, affine=False)
        conv_output_height = (
            ((self.emb_height * 2) - self.filter_size + (2 * self.padding))
            / self.stride
        ) + 1
        conv_output_width = (
            (self.emb_width-self.filter_size + (2*self.padding)) / self.stride
        ) + 1
        self.projection = torch.nn.Linear(
            32 * int(conv_output_height*conv_output_width), int(self.emb_dim)
        )
        self.non_linear = torch.nn.ReLU()

    def forward(self, s_emb, p_emb):
        batch_size = p_emb.size(0)
        s_emb_2d = s_emb[:, 1:].view(-1, 1, int(self.emb_height), int(self.emb_width))
        p_emb_2d = p_emb[:, 1:].view(-1, 1, int(self.emb_height), int(self.emb_width))
        stacked_inputs = torch.cat([s_emb_2d, p_emb_2d], 2)
        out = self.convolution(stacked_inputs)
        out = self.bn1(out)
        out = self.non_linear(out)
        out = self.feature_map_dropout(out)
        out = out.view(batch_size, -1)
        out = self.projection(out)
        out = self.projection_dropout(out)
        out = self.bn2(out)
        # once we hack
        out = self.non_linear(out)
        out = torch.cat(
            [torch.ones((batch_size, 1), device=out.device), out], 
            -1)
        return out


class ConvE(nn.Module, NeuralBinaryPredicate):
    def __init__(self,
                 num_entities,
                 num_relations,
                 embedding_dim,
                 aspect_ratio=2,
                 filter_size=3,
                 stride=1,
                 padding=0,
                 feature_map_dropout=0.2,
                 projection_dropout=0.3,
                 convolution_bias=True,
                 device='cpu',
                 **kwargs):
        super(ConvE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self._entity_embedding = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self._entity_embedding.weight)
        self._relation_embedding = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self._relation_embedding.weight)

        self._scorer = ConvEScorer(self.embedding_dim,
                                   aspect_ratio,
                                   filter_size,
                                   stride,
                                   padding,
                                   feature_map_dropout,
                                   projection_dropout,
                                   convolution_bias)

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        est_emb = self.estimte_tail_emb(head_emb, rel_emb)
        return torch.sum(est_emb * tail_emb, dim=-1)

    def score2truth_value(self, score):
        return torch.sigmoid(score)

    def estimate_tail_emb(self, head_emb, rel_emb):
        return self._scorer(head_emb, rel_emb)

    def get_relation_emb(self, relation_id_or_tensor, inv=False):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        if inv:
            pair_id = torch.div(rel_id, 2, rounding_mode='floor')
            origin_modulo_id = torch.remainder(rel_id, 2)
            inv_modulo_id_raw = origin_modulo_id + 1
            inv_modulo_id = torch.remainder(inv_modulo_id_raw, 2)
            rel_id = pair_id * 2 + inv_modulo_id
        return self._relation_embedding(rel_id)

    def get_entity_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

