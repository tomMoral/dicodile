

def get_full_shape(valid_shape, atom_shape):
    return tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_shape, atom_shape)
    ])


def get_valid_shape(sig_shape, atom_shape):
    return tuple([
        size_ax - size_atom_ax + 1
        for size_ax, size_atom_ax in zip(sig_shape, atom_shape)
    ])
