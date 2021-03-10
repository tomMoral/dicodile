from dicodile.data.images import fetch_mandrill, fetch_letters_pami


def test_fetch_mandrill():
    data = fetch_mandrill()
    assert (3, 512, 512) == data.shape


def test_fetch_letters_pami():
    X, D = fetch_letters_pami()
    assert (2321, 2004) == X.shape
    assert (4, 29, 25) == D.shape
