from dicodile.data.gait import get_gait_data


def test_get_gait():
    trial = get_gait_data()
    assert trial['Subject'] == 1
    assert trial['Trial'] == 1
    assert len(trial['data'].columns) == 16
