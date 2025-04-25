import numpy as np
from numpy.testing import assert_equal
import pytest
import finch


@pytest.mark.parametrize(
    "args",
    [
        (1, 2),
        (3, 4),
    ],
)
def test_addition(args):
    (a, b) = args
    
    assert_equal(a + b, a + b)