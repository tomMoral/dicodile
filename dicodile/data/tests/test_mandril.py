from dicodile.data.images import get_mandril


def test_fetch_mandril():
    data = get_mandril()
    assert (3, 512, 512) == data.shape
