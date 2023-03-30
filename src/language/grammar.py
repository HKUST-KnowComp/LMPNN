"""
Parse the grammar into the classes
Formula = (Formula)
        = Formula|Formula
        = Formula&Formula
        = !Formula
        = Predicate(Term,Term)
Predicate = r[number]
Term = e[number]
     = u[number]
     = l[number]
     = f[number]
"""

from .foq import Conjunction, Disjunction, Formula, Lobject, Negation, Atomic, Term


def remove_outmost_backets(lstr: str):
    if not (lstr[0] == '(' and lstr[-1] == ')'):
        return lstr

    bracket_stack = []
    for i, c in enumerate(lstr):
        if c == '(':
            bracket_stack.append(i)
        elif c == ')':
            left_bracket_index = bracket_stack.pop(-1)

    assert len(bracket_stack) == 0
    if left_bracket_index == 0:
        return lstr[1:-1]
    else:
        return lstr


def remove_brackets(lstr: str):
    new_lstr = remove_outmost_backets(lstr)
    while new_lstr != lstr:
        lstr = new_lstr
        new_lstr = remove_outmost_backets(lstr)
    return lstr


def map_term_name_to_type(name: str):
    c = name[0]
    if c == 'e':
        return Term.EXISTENTIAL, True
    elif c == 'f':
        return Term.FREE, True
    elif c == 'u':
        return Term.UNIVERSAL, True
    elif c == 's':
        return Term.SYMBOL, True
    else:
        assert name.isnumeric()
        term_id = int(name)
        return term_id, False


def parse_term(term_name):
    assert ')' not in term_name
    term_state, is_abstract = map_term_name_to_type(term_name)
    if is_abstract:
        term = Term(state=term_state, name=term_name)
    else:
        term = Term(state=Term.SYMBOL, name="symbol_by_id")
        term.entity_id_list.append(term_state)
    return term


def identify_top_binary_operator(lstr: str):
    """
    identify the top-level binary operator
    """
    _lstr = remove_brackets(lstr)
    bracket_stack = []
    for i, c in enumerate(_lstr):
        if c == '(':
            bracket_stack.append(i)
        elif c == ')':
            bracket_stack.pop(-1)
        elif c in "&|" and len(bracket_stack) == 0:
            return c, i
    return None, -1


def parse_lstr_to_lformula(lstr: str) -> Formula:
    """
    parse the string a.k.a, lstr to lobject
    """
    _lstr = remove_brackets(lstr)

    # identify top-level operator
    if _lstr[0] == '!':
        sub_lstr = _lstr[1:]
        sub_formula = parse_lstr_to_lformula(sub_lstr)
        if sub_formula.op == 'pred':
            sub_formula.negated = True
            return Negation(formula=sub_formula)

    binary_operator, binary_operator_index = identify_top_binary_operator(_lstr)

    if binary_operator_index >= 0:
        left_lstr = _lstr[:binary_operator_index]
        left_formula = parse_lstr_to_lformula(left_lstr)
        right_lstr = _lstr[binary_operator_index+1:]
        right_formula = parse_lstr_to_lformula(right_lstr)
        if binary_operator == '&':
            return Conjunction(formulas=[left_formula, right_formula])
        if binary_operator == '|':
            return Disjunction(formulas=[left_formula, right_formula])
    else:  # parse predicate
        assert _lstr[-1] == ')'
        predicate_name, right_lstr = _lstr.split('(')
        right_lstr = right_lstr[:-1]
        term1_name, term2_name = right_lstr.split(',')

        term1 = parse_term(term1_name)
        term2 = parse_term(term2_name)
        if predicate_name.isnumeric():
            predicate_id = int(predicate_name)
            predicate = Atomic(relation=f"predicate_id={predicate_id}",
                                        head=term1,
                                        tail=term2)
            predicate.relation_id_list.append(predicate_id)
        else:
            predicate = Atomic(relation=predicate_name,
                                        head=term1,
                                        tail=term2)
        return predicate
