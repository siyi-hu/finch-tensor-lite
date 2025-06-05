import numpy as np


def and_test(a, b):
    return a & b


def or_test(a, b):
    return a | b


def not_test(a):
    return not a


def ifelse(a, b, c):
    return a if c else b


def promote_min(a, b):
    cast = np.promote_types(np.result_type(a), np.result_type(b)).type
    return min(cast(a), cast(b))


def promote_max(a, b):
    cast = np.promote_types(np.result_type(a), np.result_type(b)).type
    return max(cast(a), cast(b))
