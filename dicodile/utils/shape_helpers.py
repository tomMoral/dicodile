

def get_full_support(valid_support, atom_support):
    return tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_support, atom_support)
    ])


def get_valid_support(sig_support, atom_support):
    return tuple([
        size_ax - size_atom_ax + 1
        for size_ax, size_atom_ax in zip(sig_support, atom_support)
    ])
