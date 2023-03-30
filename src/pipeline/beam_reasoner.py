import torch
import math

from src.language.foq import EFO1Query
from src.language.tnorm import Tnorm
from src.pipeline.reasoner import Reasoner
from src.structure.neural_binary_predicate import NeuralBinaryPredicate

class BEAMReasoner(Reasoner):
    """
    Gradient based Reasoner (CQD-CO) for Existential First Order (EFO) formulas.
    """
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 tnorm: Tnorm,
                 beam_size: int):
        self.beam_size = beam_size
        self.nbp = nbp
        self.tnorm: Tnorm = tnorm

        # determined during the optimization

    def initialize_with_query(self, formula: EFO1Query):
        self.formula = formula
        self.term_local_emb_dict = {
            term_name: None
            for term_name in self.formula.term_dict}

        self._last_ground_free_var_emb = {}


    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor.detach().clone()
        self.term_local_emb_dict[key].requires_grad = True

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def initialize_variable_embeddings(self):
        """
        Input args:
        Return args:
            evars: list of existential variables
            uvars: list of universal variables
            fvars: list of free variables
        """

        def check_all_var_initialized():
            return all(self.term_initialized(term_name)
                       for term_name in self.formula.term_dict)

        while not check_all_var_initialized():
            for alstr, atomic in self.formula.atomic_dict.items():
                pred_name = atomic.relation
                head_name, tail_name = atomic.head.name, atomic.tail.name

                if self.term_initialized(head_name) and not self.term_initialized(tail_name):
                    head_emb = self.get_ent_emb(
                        head_name
                    )
                    rel_emb = self.get_rel_emb(pred_name)

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    self.set_local_embedding(tail_name, tail_emb)

                elif not self.term_initialized(head_name) and self.term_initialized(tail_name):
                    tail_emb = self.get_ent_emb(
                        tail_name
                    )
                    rel_emb = self.get_rel_emb(pred_name)

                    head_emb = self.nbp.estimate_head_emb(tail_emb, rel_emb)
                    # formula.set_var_local_embedding(head_name, head_emb)
                    self.set_local_embedding(head_name, head_emb)

                else:
                    continue

        return
