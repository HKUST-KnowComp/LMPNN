import json

from ..utils.data import iter_triple_from_tsv


class KGIndex:
    """
    A class stores the maps and inverse maps
        from the entity names to entity ids
        from the relation names to relation ids
    """
    def __init__(self) -> None:
        self.map_entity_name_to_id = {}
        self.inverse_entity_id_to_name = {}
        self.map_relation_name_to_id = {}
        self.inverse_relation_id_to_name = {}

    def register_entity(self, name, eid=None):
        L = len(self.map_entity_name_to_id)
        assert L == len(self.inverse_entity_id_to_name)

        if name not in self.map_entity_name_to_id:
            if eid is None: eid = L
            else: eid = int(eid)
            self.map_entity_name_to_id[name] = eid
            self.inverse_entity_id_to_name[eid] = name
        assert len(self.inverse_entity_id_to_name) == len(
            self.map_entity_name_to_id)

    def register_relation(self, name, rid=None):
        L = len(self.map_relation_name_to_id)
        assert L == len(self.inverse_relation_id_to_name)
        if name not in self.map_relation_name_to_id:
            if rid is None: rid = L
            else: rid = int(rid)
            self.map_relation_name_to_id[name] = rid
            self.inverse_relation_id_to_name[rid] = name
        assert len(self.inverse_relation_id_to_name) == len(
            self.map_relation_name_to_id)

    def dump(self, filename):
        with open(filename, 'wt') as f:
            json.dump({'e': self.map_entity_name_to_id,
                       'r': self.map_relation_name_to_id}, f)

    @classmethod
    def load(cls, filename):
        obj = cls()
        with open(filename, 'rt') as f:
            load = json.load(f)
        obj.map_entity_name_to_id = load['e']
        obj.inverse_entity_id_to_name = {
            v: k for k, v in obj.map_entity_name_to_id.items()
        }
        obj.map_relation_name_to_id = load['r']
        obj.inverse_relation_id_to_name = {
            v: k for k, v in obj.map_relation_name_to_id.items()
        }
        return obj

    @property
    def num_entities(self):
        return len(self.map_entity_name_to_id)

    @property
    def num_relations(self):
        return len(self.map_relation_name_to_id)

    def dump_id2name(self, entity_id_file="", relation_id_file="", rel_dup=False):
        if entity_id_file:
            with open(entity_id_file, 'wt') as f:
                for eid, name in self.inverse_entity_id_to_name.items():
                    f.write(f"{eid}\t{name}\n")

        if relation_id_file:
            with open(relation_id_file, 'wt') as f:
                for rid, name in self.inverse_relation_id_to_name.items():
                    if (not rel_dup) and (rid % 2 == 1):
                        continue
                    else:
                        rid = rid // 2
                        f.write(f"{rid}\t{name}\n")