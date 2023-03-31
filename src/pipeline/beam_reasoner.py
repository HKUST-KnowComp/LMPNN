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
        self.bfs_var_name_levels = None

        
    def initialize_with_query(self, formula: EFO1Query):
        self.formula = formula
        self.term_local_emb_dict = {
            term_name: None
            for term_name in self.formula.term_dict}

        self._last_ground_free_var_emb = {}

        # determin variable ordering via a topological sort
        self.bfs_var_name_levels = self.formula.get_bfs_variable_ordering('f')


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
        return

    def estimate_variable_embeddings(self):
        """
        Actually, in beam search we don't need to estimate the "embedding"
        Instead, we estimate the fuzzy set vector of each variable
        """
        if self.bfs_var_name_levels:
            var_name_list = self.bfs_var_name_levels.pop(-1)
            for var_name in var_name_list:
                for atomic_name in self.formula.term_name2atomic_name_list[var_name]:
                    counter = 0
                    head, tail = self.formula.atomic_dict[atomic_name].get_terms():
                    if head
                                

                    assert counter == 1, "The number of initialized terms should be 1"

                    for term in self.formula.atomic_dict[atomic_name].get_terms():
                        if term.name == var_name:
                            continue
                        elif 
                            self._last_ground_free_var_emb[var_name] = self.term_local_emb_dict[term.name]

