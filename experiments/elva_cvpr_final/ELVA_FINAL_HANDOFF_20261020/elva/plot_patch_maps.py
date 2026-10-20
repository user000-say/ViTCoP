"""Plot actual retained patch IDs; no fake attention. English labels by default."""
import argparse
import json
from pathlib import Path
import numpy as np


def make_maps(ids_by_method: dict, output: str, *, grid_h=24, grid_w=24):
    if not ids_by_method:
        raise ValueError('empty ID sets')
    n = grid_h * grid_w
    from matplotlib import pyplot as plt
    methods = list(ids_by_method)
    fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods),4), squeeze=False)
    for ax, name in zip(axes[0], methods):
        ids = [int(x) for x in ids_by_method[name]]
        if len(set(ids)) != len(ids) or any(i < 0 or i >= n for i in ids):
            raise ValueError(f'{name}: IDs not unique or out of grid')
        arr = np.zeros(n, dtype=np.float32); arr[ids] = 1.
        ax.imshow(arr.reshape(grid_h, grid_w), cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'{name} ({len(ids)})')
        ax.set_xlabel('Patch column'); ax.set_ylabel('Patch row')
    fig.tight_layout();fig.savefig(output, dpi=180);plt.close(fig)
    return Path(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('json_file');p.add_argument('output')
    p.add_argument('--grid-h', type=int, default=24);p.add_argument('--grid-w', type=int, default=24)
    a=p.parse_args()
    d=json.loads(Path(a.json_file).read_text(encoding='utf-8'))
    make_maps(d,a.output,grid_h=a.grid_h,grid_w=a.grid_w)

if __name__=='__main__': main()
