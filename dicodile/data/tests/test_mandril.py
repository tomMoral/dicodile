from dicodile.data.images import fetch_mandrill

def test_fetch_mandril():
    data = fetch_mandrill()
    assert(3, 512, 512 == data.shape)
