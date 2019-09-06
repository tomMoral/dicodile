
STYLES = {
    'lgcd': {
        'color': 'C1',
        'linestyle': 'o-',
        'hatch': '//',
        'label': 'LGCD',
        'label_p': 'DiCoDiLe$_Z$'
    },
    'greedy': {
        'color': 'C0',
        'linestyle': 's-',
        'hatch': None,
        'label': 'Greedy',
        'label_p': 'Dicod'
    },
    'cyclic': {
        'color': 'C2',
        'linestyle': '^-',
        'hatch': None,
        'label': 'Cyclic',
        'label_p': 'Cyclic'
    },
}


def get_style(name, *keys, parallel=False):
    all_style = STYLES[name]
    style = {
        'label': all_style['label_p'] if parallel else all_style['label'],
        'color': all_style['color']
    }
    for k in keys:
        style[k] = all_style[k]
    return style
