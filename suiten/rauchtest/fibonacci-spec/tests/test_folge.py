import time

import pytest

from folge import fibonacci


def test_anfang():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_zehntes_glied():
    assert fibonacci(10) == 55


def test_negativ_wirft():
    with pytest.raises(ValueError):
        fibonacci(-1)


def test_schnell_genug():
    """Die naive Rekursion braucht fuer n=30 rund eine Sekunde - Absicht."""
    t0 = time.perf_counter()
    assert fibonacci(30) == 832040
    assert time.perf_counter() - t0 < 1.0
