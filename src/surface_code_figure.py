"""
Figure 2.1 — Surface code stabilizer tiling (surface code, no periodic BC).

Convention matching panqec / later simulation figures:
  - Data qubits on EDGES of the lattice (half-integer coords)
  - X-type (vertex/star) stabilizers: pink diamonds at lattice vertices
  - Z-type (face/plaquette) stabilizers: teal squares at lattice faces

This is the standard unrotated surface code representation.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D

X_COLOR = '#CC79A7'   # pink  — X-type vertex/star stabilizers
Z_COLOR = '#009E73'   # teal  — Z-type face/plaquette stabilizers
Q_COLOR = '#333333'   # dark grey — data qubits


def fig_surface_code(n=4, save=True):
    """
    Parameters
    ----------
    n : int
        Number of lattice cells per side.
        Produces (n+1)² vertices, n² faces, and edge qubits between them.
    """
    plt.style.use('seaborn-v0_8-white')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rcParams.update({'legend.fontsize': 13, 'font.size': 13})

    fig, ax = plt.subplots(figsize=(6, 7))

    # ── Thin lattice grid lines ───────────────────────────────────────────────
    for i in range(n + 1):
        ax.axhline(i, color='gray', lw=0.7, alpha=0.5, zorder=4.5)
        ax.axvline(i, color='gray', lw=0.7, alpha=0.5, zorder=4.5)

    # ── Z-type stabilizers: teal squares at each face ─────────────────────────
    # Light fill + teal border so faces are visible without dominating
    for i in range(n):
        for j in range(n):
            sq = np.array([[i, j], [i+1, j], [i+1, j+1], [i, j+1]], float)
            ax.add_patch(Polygon(sq, closed=True, zorder=2,
                                 facecolor=Z_COLOR, alpha=0.22,
                                 edgecolor='white', linewidth=2.0))

    # ── X-type stabilizers: pink diamonds at each vertex ─────────────────────
    # Diamond corners reach the 4 adjacent edge-qubit positions (±0.5 from vertex).
    # Clipped to axes bounds so boundary diamonds don't produce artefacts.
    for i in range(n + 1):
        for j in range(n + 1):
            diamond = np.array([
                [i - 0.5, j],
                [i,        j + 0.5],
                [i + 0.5,  j],
                [i,        j - 0.5],
            ])
            bg = Polygon(diamond, closed=True, zorder=3,
                         facecolor='white', alpha=1.0,
                         edgecolor='none')
            bg.set_clip_path(ax.patch)
            ax.add_patch(bg)
            p = Polygon(diamond, closed=True, zorder=4,
                        facecolor=X_COLOR, alpha=0.45,
                        edgecolor=X_COLOR, linewidth=1.2)
            p.set_clip_path(ax.patch)
            ax.add_patch(p)

    # ── Highlighted vertex (X-type / star) ───────────────────────────────────
    hv = (1, 2)
    for dx, dy in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
        x1, y1 = hv[0] + dx, hv[1] + dy
        if 0 <= x1 <= n and 0 <= y1 <= n:
            ax.plot([hv[0], x1], [hv[1], y1],
                    color=X_COLOR, lw=3.5, zorder=4.8, solid_capstyle='round')

    # ── Highlighted plaquette (Z-type / face) ─────────────────────────────────
    hp = (2, 1)  # bottom-left corner of face
    corners = [(hp[0], hp[1]), (hp[0]+1, hp[1]),
               (hp[0]+1, hp[1]+1), (hp[0], hp[1]+1)]
    for k in range(4):
        x0, y0 = corners[k]
        x1, y1 = corners[(k+1) % 4]
        ax.plot([x0, x1], [y0, y1],
                color=Z_COLOR, lw=3.5, zorder=4.8, solid_capstyle='round')

    # ── Ancilla qubits ────────────────────────────────────────────────────────
    # Z-type ancilla at face centres
    fz_x = [i + 0.5 for i in range(n) for j in range(n)]
    fz_y = [j + 0.5 for i in range(n) for j in range(n)]
    ax.scatter(fz_x, fz_y, c=Z_COLOR, s=65, zorder=5)

    # X-type ancilla at vertex positions
    vx_x = [float(i) for i in range(n + 1) for j in range(n + 1)]
    vx_y = [float(j) for i in range(n + 1) for j in range(n + 1)]
    ax.scatter(vx_x, vx_y, c=X_COLOR, s=65, zorder=5)

    # ── Data qubits on edges ──────────────────────────────────────────────────
    hx = [i + 0.5 for i in range(n) for j in range(n + 1)]
    hy = [float(j) for i in range(n) for j in range(n + 1)]
    vex = [float(i) for i in range(n + 1) for j in range(n)]
    vey = [j + 0.5 for i in range(n + 1) for j in range(n)]
    ax.scatter(hx + vex, hy + vey, c=Q_COLOR, s=65, zorder=6)

    # ── Legend (below axes so it never overlaps data) ─────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=Q_COLOR,
               markersize=9, label='data qubit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=X_COLOR,
               markersize=9, label='$X$-type ancilla'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=Z_COLOR,
               markersize=9, label='$Z$-type ancilla'),
        Patch(facecolor=X_COLOR, alpha=0.55,
              label='$X$-type stabilizer (vertex)'),
        Patch(facecolor=Z_COLOR, alpha=0.55,
              label='$Z$-type stabilizer (face)'),
    ]
    ax.legend(handles=legend_elements,
              loc='upper center', bbox_to_anchor=(0.5, -0.04),
              ncol=2, frameon=True, framealpha=0.95)

    ax.set_aspect('equal')
    ax.set_xlim(-0.55, n + 0.55)
    ax.set_ylim(-0.55, n + 0.55)
    ax.axis('off')

    plt.tight_layout()

    if save:
        out = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'figures',
                         'surface_code_stabilizers.pdf'))
        fig.savefig(out, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    fig_surface_code(n=4, save=True)
