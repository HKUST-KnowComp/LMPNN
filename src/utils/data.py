from typing import Union, List
from itertools import chain
import json
from random import shuffle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.language.foq import EFO1Query
from src.language.grammar import parse_lstr_to_lformula


def _iter_triple_from_tsv(triple_file, to_int, check_size):
    with open(triple_file, 'rt') as f:
        for line in f.readlines():
            triple = line.strip().split()
            if check_size:
                assert len(triple) == check_size
            if to_int:
                triple = [int(t) for t in triple]
            yield triple


def iter_triple_from_tsv(triple_files, to_int: bool=True, check_size: int=3):
    if isinstance(triple_files, list):
        return chain(*[iter_triple_from_tsv(tfile) for tfile in triple_files])
    elif isinstance(triple_files, str):
        return _iter_triple_from_tsv(triple_files, to_int, check_size)
    else:
        raise NotImplementedError("invalid input of triple files")


def tensorize_batch_entities(
        entities: Union[List[int], List[List[int]], torch.Tensor],
        device) -> torch.Tensor:
    """
    convert the entities into the tensor formulation
    in the shape of [batch_size, num_entities]
    we interprete three cases
    1. List[int] batch size = 1
    2. List[List[int]], each inner list is a sample
    3. torch.Tensor in shape [batch_size, num_entities]
    """
    if isinstance(entities, list):
        if isinstance(entities[0], int):
            # in this case, batch size = 1
            entity_tensor = torch.tensor(
                entities, device=device).reshape(-1, 1)
        elif isinstance(entities[0], list):
            # in this case, batch size = len(entities)
            assert isinstance(entities[0][0], int)
            entity_tensor = torch.tensor(
                entities, device=device).reshape(len(entities), -1)
        else:
            raise NotImplementedError(
                "higher order nested list is not supported")
    elif isinstance(entities, torch.Tensor):
        assert entities.dim() == 2
        entity_tensor = entities.to(device)
    else:
        raise NotImplementedError("unsupported input entities type")
    return entity_tensor


class RaggedBatch:
    def __init__(self, flatten, sizes):
        self.flatten = flatten
        self.sizes = sizes

    def run_ops_on_flatten(self, opfunc):
        return RaggedBatch(
            flatten=opfunc(self.flatten),
            sizes=self.sizes)

    def to_dense_matrix(self, padding_value):
        # split the first axis of the flattened Tensor by sizes
        flatten_sliced = torch.split(
            self.flatten, split_size_or_sections=self.sizes, dim=0)
        dense_matrix = pad_sequence(
            flatten_sliced, batch_first=True, padding_value=padding_value)
        # if the self.flattened is of shape [L, *]
        # then dense_matrix is of shape [batch_size, max_of_self.sizes, *]
        return dense_matrix



class QAACollator:
    def __init__(self, lstr):
        self.lstr = lstr

    def __call__(self, batch_input):
        lformula = parse_lstr_to_lformula(self.lstr)
        query = EFO1Query(lformula)
        for rsdict, easy_ans, hard_ans in batch_input:
            query.append_qa_instances(rsdict, easy_ans, hard_ans)
        return query


class QueryAnsweringSeqDataLoader:
    def __init__(self, qaafile, target_lstr=None, size_limit=-1, **dataloader_kwargs) -> None:
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_iterator = {}
        for lstr, qaa in self.lstr_qaa.items():
            _lstr = parse_lstr_to_lformula(lstr).lstr
            if target_lstr:
                if _lstr not in target_lstr:
                    print(lstr, "query not selected, continue")
                    continue
            if not qaa:
                print(_lstr, "query type is empty, continue")
                continue

            if size_limit > 0:
                qaa = qaa[:size_limit]

            print(f"query {_lstr} of size {len(qaa)} loaded from {qaafile}")
            self.lstr_iterator[_lstr] = DataLoader(qaa,
                collate_fn=QAACollator(_lstr),
                **self.dataloader_kwargs)


    def get_fof_list(self):
        batch_buffer = []
        for _, iterator in self.lstr_iterator.items():
            for batch in iterator:
                batch_buffer.append(batch)
        shuffle(batch_buffer)
        return batch_buffer
