import os
import torch
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import ComplEx

kgs = ['FB15k-237', 'FB15k', 'NELL']

if __name__ == "__main__":

    for kgname in kgs:

        kgidx = KGIndex.load(os.path.join('data', kgname + '-betae', 'kgindex.json'))

        for par, dirs, files in os.walk('pretrain/models'):
            for fname in files:
                if fname.endswith('.pt') and fname.startswith(kgname):
                    # the desired objective
                    terms = fname.split('-')
                    rank = None
                    epoch = None
                    for i, t in enumerate(terms):
                        if t == 'rank':
                            rank = int(terms[i+1])
                        if t == 'epoch':
                            epoch = int(terms[i+1])
                    assert rank
                    assert epoch

                    my_model = ComplEx(num_entities=kgidx.num_entities,
                            num_relations=kgidx.num_relations,
                            embedding_dim=rank)

                    state_dict = torch.load(os.path.join(par, fname), map_location='cpu')

                    my_model._entity_embedding.weight.data = state_dict['model_state_dict']['embeddings.0.weight']
                    my_model._relation_embedding.weight.data = state_dict['model_state_dict']['embeddings.1.weight']

                    torch.save(my_model.state_dict(), os.path.join('pretrain', 'complex', fname))

                    pass
