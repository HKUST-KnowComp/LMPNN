
import torch
from torch import nn

from .neural_binary_predicate import NeuralBinaryPredicate

class TransE(nn.Module, NeuralBinaryPredicate):
    def __init__(self, num_entities, num_relations, embedding_dim, p, margin, scale, device, **kwargs):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.scale = margin
        self.scale = scale
        self.p = p
        self._entity_embedding = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self._entity_embedding.weight)
        self._relation_embedding = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self._relation_embedding.weight)

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        return - torch.norm(head_emb + rel_emb - tail_emb, p=self.p, dim=-1)

    def score2truth_value(self, score):
        return torch.sigmoid(self.scale + score * self.scale)

    def estimate_tail_emb(self, head_emb, rel_emb):
        return head_emb + rel_emb

    def estimate_head_emb(self, tail_emb, rel_emb):
        return tail_emb - rel_emb

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

    def get_tail_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)
