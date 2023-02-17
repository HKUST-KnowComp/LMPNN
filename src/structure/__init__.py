def get_nbp_class(name):
    if name.lower() == 'transe':
        from .nbp_transe import TransE
        return TransE
    if name.lower() == 'swtranse':
        from .nbp_swtranse import SWTransE
        return SWTransE
    if name.lower() == 'complex':
        from .nbp_complex import ComplEx
        return ComplEx
    if name.lower() == 'rotate':
        from .nbp_rotate import RotatE
        return RotatE
    if name.lower() == 'distmult':
        from .nbp_distmult import DistMult
        return DistMult
    if name.lower() == 'conve':
        from .nbp_conve import ConvE
        return ConvE
    if name.lower() == 'rescal':
        from .nbp_rescal import RESCAL
        return RESCAL
