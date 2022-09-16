from pathlib import Path

from dicodile.config import get_data_home


def test_dicodile_home(monkeypatch):
    _set_env(monkeypatch, {
        "DICODILE_DATA_HOME": "/home/unittest/dicodile"
    })
    assert get_data_home() == Path("/home/unittest/dicodile/dicodile")


def test_XDG_DATA_home(monkeypatch):
    _set_env(monkeypatch, {
        "DICODILE_DATA_HOME":  None,
        "XDG_DATA_HOME": "/home/unittest/data"
    })
    assert get_data_home() == Path("/home/unittest/data/dicodile")


def test_default_home(monkeypatch):
    _set_env(monkeypatch, {
        "HOME": "/home/default",
        "DICODILE_DATA_HOME":  None,
        "XDG_DATA_HOME": None,
    })
    assert get_data_home() == Path("/home/default/data/dicodile")


def test_dicodile_home_has_priority_over_xdg_data_home(monkeypatch):
    _set_env(monkeypatch, {
        "DICODILE_DATA_HOME": "/home/unittest/dicodile",
        "XDG_DATA_HOME": "/home/unittest/data"
    })
    assert get_data_home() == Path("/home/unittest/dicodile/dicodile")


def _set_env(monkeypatch, d):
    for k, v in d.items():
        if v is not None:
            monkeypatch.setenv(k, v)
        else:
            monkeypatch.delenv(k, raising=False)
