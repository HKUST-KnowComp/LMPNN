from abc import abstractmethod

import torch


class NeuralBinaryPredicate:
    num_entities: int
    num_relations: int
    device: torch.device
    scale: float

    @abstractmethod
    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        This method computes the score for the triple given the head, tail and
        relation embedding. The higher score means more likely to be a predicate.
        Inputs:
            Three embeddings are in the shape [..., embed_dim]
        Returns:
            The tensor of scores in the shape [...]
        """
        pass

    @abstractmethod
    def score2truth_value(self, score):
        pass

    @abstractmethod
    def estimate_tail_emb(self, head_emb, rel_emb):
        pass

    @abstractmethod
    def estimate_head_emb(self, tail_emb, rel_emb):
        pass

    @abstractmethod
    def estiamte_rel_emb(self, head_emb, tail_emb):
        pass

    @abstractmethod
    def get_relation_emb(self, relation_id_or_tensor, inv=False):
        pass

    @abstractmethod
    def get_entity_emb(self, entity_id_or_tensor):
        pass

    @property
    def entity_embedding(self) -> torch.Tensor:
        pass

    @property
    def relation_embedding(self) -> torch.Tensor:
        pass

    @classmethod
    def create(cls, device, **kwargs):
        obj = cls(device=device, **kwargs)
        obj = obj.to(device)
        return obj

    def batch_predicate_score(self,
                              triple_tensor: torch.Tensor) -> torch.Tensor:
        """
        This method computes the scores for the triple. triple tensors the
        shape of [..., 3]
        It returns the same size of predicate scores.
        """
        if isinstance(triple_tensor, list):
            assert len(triple_tensor) == 3
            head_id_ten, rel_id_ten, tail_id_ten = triple_tensor
        else:
            head_id_ten, rel_id_ten, tail_id_ten = torch.split(
                triple_tensor, 1, dim=-1)
        head_emb = self._entity_embedding(head_id_ten)
        rel_emb = self._relation_embedding(rel_id_ten)
        tail_emb = self._entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)

    def get_all_entity_rankings(self, batch_embedding_input, eval_batch_size=16, score='cos'):
        batch_size = batch_embedding_input.size(0)
        begin = 0
        entity_ranking_list = []
        for begin in range(0, batch_size, eval_batch_size):
            end = begin + eval_batch_size
            eval_batch_embedding_input = batch_embedding_input[begin: end]
            eval_batch_embedding_input = eval_batch_embedding_input.unsqueeze(-2)
            # batch_size, all_candidates
            # ranking score should be the higher the better
            # ranking_score[entity_id] = the score of {entity_id}
            # ranking_score = self.entity_pair_scoring(eval_batch_embedding_input, self.entity_embedding)
            if score == 'cos':
                ranking_score = torch.cosine_similarity(eval_batch_embedding_input, self.entity_embedding, dim=-1)
            else:
                ranking_score = - torch.norm(eval_batch_embedding_input - self.entity_embedding, dim=-1)
            # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
            ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
            # entity_rankings[entity_id] = {rankings} of the entity
            entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
            entity_ranking_list.append(entity_rankings)

        batch_entity_rankings = torch.cat(entity_ranking_list, dim=0)
        return batch_entity_rankings

    
    def get_all_entity_scores(self, batch_embedding_input, eval_batch_size=16):
        batch_size = batch_embedding_input.size(0)
        begin = 0
        entity_scoring_list = []
        for begin in range(0, batch_size, eval_batch_size):
            end = begin + eval_batch_size
            eval_batch_embedding_input = batch_embedding_input[begin: end]
            eval_batch_embedding_input = eval_batch_embedding_input.unsqueeze(-2)
            # batch_size, all_candidates
            # ranking score should be the higher the better
            # ranking_score[entity_id] = the score of {entity_id}
            ranking_score = self.entity_pair_scoring(eval_batch_embedding_input, self.entity_embedding)
            entity_scoring_list.append(ranking_score)

        batch_entity_scoring = torch.cat(entity_scoring_list, dim=0)
        return batch_entity_scoring
