import operator


def and_test(a, b) -> bool:
    return operator.truth(a) and operator.truth(b)


def or_test(a, b) -> bool:
    return operator.truth(a) or operator.truth(b)


def not_test(a) -> bool:
    return not operator.truth(a)


# def ifelse:
#     return expr
