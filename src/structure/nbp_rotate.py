import torch
from torch import nn

from .neural_binary_predicate import NeuralBinaryPredicate


class RotatE(NeuralBinaryPredicate, nn.Module):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 scale: float = 1,
                 init_size: float = 1e-3,
                 device = 'cpu', **kwargs):
        super(RotatE, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.scale = scale

        self._entity_embedding = nn.Embedding(num_entities, 2*embedding_dim)
        self._entity_embedding.weight.data *= init_size

        self._relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self._relation_embedding.weight.data *= init_size

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight

    @property
    def relation_embedding(self):
        return self._relation_embedding.weight


    def embedding_score(self, head_emb, rel_emb, tail_emb):
        # lhs = head_emb[..., :self.rank], head_emb[..., self.rank:]
        # rel = rel_emb[..., :self.rank],  rel_emb[..., self.rank:]
        # rhs = tail_emb[..., :self.rank], tail_emb[..., self.rank:]
        # return torch.sum(
        #     (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
        #     (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
        #     dim=-1)
        est_tail = self.estimate_tail_emb(head_emb, rel_emb)
        return self.entity_pair_scoring(est_tail, tail_emb)

    def score2truth_value(self, score):
        return torch.sigmoid(score / self.scale)

    def estimate_tail_emb(self, head_emb, rel_emb):
        lhs = head_emb[:, :self.embedding_dim], head_emb[:, self.embedding_dim:]
        rel = torch.cos(rel_emb), torch.sin(rel_emb)

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def estimate_head_emb(self, tail_emb, rel_emb):
        rhs = tail_emb[:, :self.embedding_dim], tail_emb[:, self.embedding_dim:]
        rel = torch.cos(rel_emb), torch.sin(rel_emb)

        return torch.cat([
            rhs[0] * rel[0] - rhs[1] * rel[1],
            rhs[0] * rel[1] + rhs[1] * rel[0]
        ], 1)

    # def estiamte_rel_emb(self, head_emb, tail_emb):
    #     lhs = head_emb[:, :self.embedding_dim], head_emb[:, self.embedding_dim:]
    #     rhs = tail_emb[:, :self.embedding_dim], tail_emb[:, self.embedding_dim:]

    #     return torch.cat([
    #         lhs[0] * rhs[0] + lhs[1] * rhs[1],
    #         lhs[0] * rhs[1] - lhs[1] * rhs[0]
    #     ], 1)

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

    def entity_pair_scoring(self, emb1, emb2):
        scores = torch.norm(emb1 - emb2, dim=-1)
        return scores

    def get_random_entity_embed(self, batch_size):
        return torch.normal(0, 1e-3, (batch_size, self.embedding_dim * 2), device=self.device, requires_grad=True)
