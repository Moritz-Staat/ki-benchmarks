from rechner import addiere, multipliziere


def test_addiere():
    assert addiere(2, 3) == 5
    assert addiere(-1, 1) == 0


def test_multipliziere_bleibt_heil():
    """Ein Bugfix, der eine andere Funktion kaputtmacht, ist kein Bugfix."""
    assert multipliziere(3, 4) == 12
