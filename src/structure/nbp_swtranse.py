
import torch
from torch import nn

from .neural_binary_predicate import NeuralBinaryPredicate


class SWTransE(nn.Module, NeuralBinaryPredicate):
    def __init__(self, num_entities, num_relations, embedding_dim, num_particles, p, margin, scale, device, **kwargs):
        super(SWTransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_particles = num_particles
        self.device = device
        self.scale = margin
        self.scale = scale
        self.p = p
        self._entity_embedding = nn.Embedding(
            num_entities, embedding_dim * num_particles)
        nn.init.xavier_uniform_(self._entity_embedding.weight)
        self._relation_embedding = nn.Embedding(
            num_relations, embedding_dim)
        nn.init.xavier_uniform_(self._relation_embedding.weight)

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        head_emb_shape = head_emb.shape[:-1] + \
            (self.embedding_dim, self.num_particles)
        head_particles = head_emb.view(head_emb_shape)

        rel_emb = rel_emb.unsqueeze(-1)
        est_particles = head_particles + rel_emb

        tail_emb_shape = tail_emb.shape[:-1] + \
            (self.embedding_dim, self.num_particles)
        tail_particles = tail_emb.view(tail_emb_shape)

        sort_est_particles, _ = torch.sort(est_particles, dim=-1)
        sort_tail_particles, _ = torch.sort(tail_particles, dim=-1)

        dist = torch.sum(
            torch.norm(sort_est_particles-sort_tail_particles, p=self.p, dim=-1),
            -1)
        return - dist

    def score2truth_value(self, score):
        return torch.sigmoid(self.scale + score * self.scale)

    def estimate_tail_emb(self, head_emb, rel_emb):
        return head_emb + rel_emb

    def estimate_head_emb(self, tail_emb, rel_emb):
        return tail_emb - rel_emb

    def estiamte_rel_emb(self, head_emb, tail_emb):
        return tail_emb - head_emb

    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self._relation_embedding(rel_id)

    def get_entity_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_tail_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_all_entity_rankings(self, batch_embedding_input):
        batch_embedding_input = batch_embedding_input.unsqueeze(-2)
        # batch_size, all_candidates
        # ranking score should be the higher the better
        # ranking_score[entity_id] = the score of {entity_id}
        ranking_score = - \
            torch.norm(batch_embedding_input -
                       self.entity_embedding, p=self.p, dim=-1)
        # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
        ranked_entity_ids = torch.argsort(
            ranking_score, dim=-1, descending=True)
        # entity_rankings[entity_id] = {rankings} of the entity
        entity_rankings = torch.argsort(
            ranked_entity_ids, dim=-1, descending=False)
        return entity_rankings
