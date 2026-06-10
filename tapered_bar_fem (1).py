"""
=============================================================
  FEM Analysis — Circular Tapered Bar
  Method  : Direct Stiffness (Spring) Method
  Elements: n uniform bar elements (midpoint diameter)
  Author  : FEA Learning Program
=============================================================

Problem:
  - Bar fixed at left end (node 1)
  - Axial load P applied at free end (node n+1)
  - Diameter varies linearly: d1 (fixed end) → d2 (free end)
  - Young's modulus E

Step-by-step:
  1. Compute element midpoint diameter & area
  2. Compute element stiffness ke = E*Ae/h
  3. Assemble global K  (tridiagonal)
  4. Apply BC  (u1 = 0)
  5. Solve reduced system
  6. Recover reactions & element stresses
  7. Compare with exact analytical solution
"""

import numpy as np
import sys

# ── optional matplotlib for plotting ──────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ═══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_float(prompt, positive=True):
    """Read a positive float from the user."""
    while True:
        try:
            val = float(input(prompt))
            if positive and val <= 0:
                print("  ✗  Value must be > 0. Try again.")
                continue
            return val
        except ValueError:
            print("  ✗  Please enter a valid number.")


def get_int(prompt, min_val=1):
    """Read an integer >= min_val from the user."""
    while True:
        try:
            val = int(input(prompt))
            if val < min_val:
                print(f"  ✗  Value must be >= {min_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  ✗  Please enter a valid integer.")


def separator(char="─", width=62):
    print(char * width)


def section(title):
    separator("═")
    print(f"  {title}")
    separator("═")


# ═══════════════════════════════════════════════════════════
#  STEP 0 — INPUT
# ═══════════════════════════════════════════════════════════

def get_inputs():
    section("INPUT PARAMETERS")
    print()
    print("  Enter bar properties (consistent units, e.g. N, mm, MPa)")
    print()

    L  = get_float("  Bar length          L  [mm]  : ")
    d1 = get_float("  Diameter at x=0     d1 [mm]  : ")
    d2 = get_float("  Diameter at x=L     d2 [mm]  : ")
    E  = get_float("  Young's modulus     E  [MPa]  : ")
    P  = get_float("  Axial load at tip   P  [N]   : ")
    n  = get_int  ("  Number of elements  n        : ")

    print()
    print(f"  ✔  L={L} mm | d1={d1} mm | d2={d2} mm | E={E} MPa | P={P} N | n={n}")
    return L, d1, d2, E, P, n


# ═══════════════════════════════════════════════════════════
#  STEP 1 — ELEMENT GEOMETRY
# ═══════════════════════════════════════════════════════════

def compute_element_properties(n, L, d1, d2, E):
    """
    For each element e (0-indexed):
      - midpoint position  x_mid = (e + 0.5) * h
      - diameter at mid    d_mid = d1 + (d2-d1)*x_mid/L
      - area               A_e   = pi/4 * d_mid^2
      - stiffness          k_e   = E * A_e / h
    Returns arrays: h, x_nodes, d_nodes, x_mid, d_mid, A_mid, k
    """
    h       = L / n
    x_nodes = np.array([i * h          for i in range(n + 1)])
    d_nodes = d1 + (d2 - d1) * x_nodes / L

    x_mid   = np.array([(e + 0.5) * h  for e in range(n)])
    d_mid   = d1 + (d2 - d1) * x_mid / L
    A_mid   = np.pi / 4.0 * d_mid ** 2
    k       = E * A_mid / h

    return h, x_nodes, d_nodes, x_mid, d_mid, A_mid, k


def print_element_table(n, h, x_mid, d_mid, A_mid, k):
    section("STEP 1 — ELEMENT PROPERTIES  (midpoint diameter)")
    print(f"  Element length  h = L/n = {h:.4f} mm")
    print()
    print(f"  {'Elem':>5}  {'Nodes':>8}  {'x_mid(mm)':>10}  "
          f"{'d_mid(mm)':>10}  {'A_mid(mm²)':>12}  {'k_e(N/mm)':>14}")
    separator()
    for e in range(n):
        print(f"  {e+1:>5}  {e+1:>4}→{e+2:<3}  {x_mid[e]:>10.4f}  "
              f"{d_mid[e]:>10.4f}  {A_mid[e]:>12.4f}  {k[e]:>14.4f}")


# ═══════════════════════════════════════════════════════════
#  STEP 2 — ASSEMBLE GLOBAL K
# ═══════════════════════════════════════════════════════════

def assemble_global_K(n, k):
    """
    Assemble (n+1)×(n+1) global stiffness matrix.

    Assembly rules:
      K[e,   e  ] += +k_e       (diagonal,    node e)
      K[e,   e+1] += -k_e       (off-diagonal)
      K[e+1, e  ] += -k_e       (off-diagonal, symmetric)
      K[e+1, e+1] += +k_e       (diagonal,    node e+1)
    """
    size = n + 1
    K = np.zeros((size, size))

    for e in range(n):
        i, j = e, e + 1          # global node indices (0-based)
        K[i, i] += k[e]
        K[i, j] -= k[e]
        K[j, i] -= k[e]
        K[j, j] += k[e]

    return K


def print_global_K(K, n):
    section("STEP 2 — GLOBAL STIFFNESS MATRIX  K  (N/mm)")
    size = n + 1
    print(f"  Size: {size} × {size}   (Tridiagonal | Symmetric | Singular before BC)")
    print()

    # header
    hdr = "       " + "".join(f"  {'u'+str(i+1):>10}" for i in range(size))
    print(hdr)
    separator()
    for i in range(size):
        row = f"  f{i+1:>2} |"
        for j in range(size):
            row += f"  {K[i,j]:>10.2f}"
        print(row)

    print()
    print("  Pattern:  diagonal K[i,i] = k_{i-1} + k_i")
    print("            off-diag K[i,j] = -k_e  (connecting element)")


# ═══════════════════════════════════════════════════════════
#  STEP 3 — APPLY BC & SOLVE
# ═══════════════════════════════════════════════════════════

def apply_bc_and_solve(K, P, n):
    """
    BC: u1 = 0  →  remove row 0 and col 0.
    Load vector f: zero everywhere except last DOF = P.

    Returns full displacement vector u (length n+1),
    with u[0] = 0 prepended.
    """
    # Build full force vector
    f_full = np.zeros(n + 1)
    f_full[-1] = P

    # Reduced system: remove row 0 and col 0
    K_red = K[1:, 1:]
    f_red = f_full[1:]

    # Solve
    u_red = np.linalg.solve(K_red, f_red)

    # Prepend the fixed DOF
    u_full = np.concatenate(([0.0], u_red))
    return u_full, f_full


def print_reduced_system(K, P, n):
    section("STEP 3 — APPLY BC  (u₁=0)  →  Reduced System")
    K_red = K[1:, 1:]
    size  = n

    print(f"  Remove row 1 & col 1  →  {size}×{size} reduced system")
    print()

    hdr = "       " + "".join(f"  {'u'+str(i+2):>10}" for i in range(size))
    print(hdr)
    separator()
    for i in range(size):
        row = f"  f{i+2:>2} |"
        for j in range(size):
            row += f"  {K_red[i,j]:>10.2f}"
        f_val = P if i == size - 1 else 0.0
        row  += f"    =  {f_val:>10.2f}"
        print(row)


# ═══════════════════════════════════════════════════════════
#  STEP 4 — RESULTS
# ═══════════════════════════════════════════════════════════

def compute_element_stress(n, u, k, A_mid, E, h):
    """σ_e = E * (u_{e+1} - u_e) / h  for each element."""
    stress = np.array([E * (u[e+1] - u[e]) / h for e in range(n)])
    force  = stress * A_mid
    return stress, force


def exact_solution(P, L, E, d1, d2):
    """δ = 4PL / (π E d1 d2)"""
    return 4.0 * P * L / (np.pi * E * d1 * d2)


def print_results(n, u, x_nodes, stress, force, A_mid, P, L, E, d1, d2):
    # ── Displacements ──
    section("STEP 4a — NODAL DISPLACEMENTS")
    print(f"  {'Node':>5}  {'x (mm)':>10}  {'u (mm)':>16}")
    separator()
    for i in range(n + 1):
        marker = "  ← fixed" if i == 0 else ("  ← FREE END ★" if i == n else "")
        print(f"  {i+1:>5}  {x_nodes[i]:>10.4f}  {u[i]:>16.8f}{marker}")

    # ── Element stresses ──
    print()
    section("STEP 4b — ELEMENT STRESSES & FORCES")
    print(f"  {'Elem':>5}  {'σ (MPa)':>14}  {'F (N)':>14}  {'A_mid (mm²)':>12}")
    separator()
    for e in range(n):
        print(f"  {e+1:>5}  {stress[e]:>14.6f}  {force[e]:>14.4f}  {A_mid[e]:>12.4f}")

    # ── Reaction force ──
    R = -P    # equilibrium
    print()
    print(f"  Reaction at fixed end:  R = {R:.4f} N   (= -P ✓)")

    # ── Comparison with exact ──
    delta_exact = exact_solution(P, L, E, d1, d2)
    delta_fea   = u[n]
    error_pct   = abs(delta_fea - delta_exact) / delta_exact * 100

    print()
    section("STEP 5 — COMPARISON WITH EXACT SOLUTION")
    print(f"  Exact  δ = 4PL/(πEd₁d₂) = {delta_exact:.8f} mm")
    print(f"  FEA    δ = {delta_fea:.8f} mm")
    print(f"  Error  = {error_pct:.4f} %")
    if error_pct < 0.5:
        print(f"  ✔  Excellent accuracy with n={n} elements!")
    elif error_pct < 2.0:
        print(f"  ✔  Good accuracy. Increase n for better results.")
    else:
        print(f"  ⚠  Try increasing n for better accuracy.")


# ═══════════════════════════════════════════════════════════
#  STEP 5 — PLOT
# ═══════════════════════════════════════════════════════════

def plot_results(n, L, d1, d2, x_nodes, u, k, A_mid, E, h, P):
    if not HAS_PLOT:
        print("\n  (matplotlib not found — skipping plots)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"FEM — Circular Tapered Bar   (n={n} elements)", fontsize=14, fontweight='bold')

    # ── colour palette ──
    C_FEA   = "#4C8EFF"
    C_EXACT = "#E85D24"
    C_BAR   = "#4C8EFF"

    # ─── Plot 1: Tapered bar geometry ───────────────────────
    ax = axes[0, 0]
    ax.set_title("Bar Geometry & Element Division", fontsize=11)
    x_e = np.linspace(0, L, 300)
    d_e = d1 + (d2 - d1) * x_e / L
    ax.fill_between(x_e,  d_e / 2, -d_e / 2, alpha=0.25, color=C_BAR, label='Bar profile')
    ax.plot(x_e,  d_e / 2, color=C_BAR, linewidth=1.5)
    ax.plot(x_e, -d_e / 2, color=C_BAR, linewidth=1.5)
    for e in range(n):
        xe = e * h
        ax.axvline(xe, color='gray', linewidth=0.7, linestyle='--', alpha=0.6)
        xm = (e + 0.5) * h
        dm = d1 + (d2 - d1) * xm / L
        ax.plot([xm], [0], 'o', color=C_EXACT, markersize=5, zorder=5)
        ax.vlines(xm, -dm/2, dm/2, color=C_EXACT, linewidth=1.2, alpha=0.7)
    ax.axvline(L, color='gray', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.axvline(0, color='black', linewidth=3)
    ax.annotate('', xy=(L + L*0.07, 0), xytext=(L, 0),
                arrowprops=dict(arrowstyle='->', color=C_EXACT, lw=2))
    ax.text(L + L*0.08, 0, 'P', color=C_EXACT, fontsize=12, va='center', fontweight='bold')
    ax.set_xlabel("x (mm)"); ax.set_ylabel("Radius (mm)")
    ax.legend(handles=[
        mpatches.Patch(color=C_BAR,   label='Bar profile'),
        plt.Line2D([0],[0], marker='o', color=C_EXACT, label='Midpoint (equivalent d)', linewidth=0)
    ], fontsize=8)
    ax.set_xlim(-L*0.05, L*1.18)
    ax.grid(True, alpha=0.3)

    # ─── Plot 2: Displacement ────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Displacement u(x)", fontsize=11)
    # Exact curve
    x_fine = np.linspace(0, L, 400)
    u_exact = np.zeros(400)
    for idx, xx in enumerate(x_fine):
        dx = d1 + (d2 - d1) * xx / L
        if abs(d2 - d1) < 1e-10:
            u_exact[idx] = P * xx / (E * np.pi / 4 * d1**2)
        else:
            u_exact[idx] = (4 * P * L) / (np.pi * E * (d2 - d1)) * (1/d1 - 1/dx)
    ax.plot(x_fine, u_exact, color=C_EXACT, linewidth=2, linestyle='--', label='Exact')
    ax.plot(x_nodes, u,      color=C_FEA,   linewidth=2, marker='o', markersize=6, label=f'FEA (n={n})')
    ax.set_xlabel("x (mm)"); ax.set_ylabel("u (mm)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ─── Plot 3: Element stiffness ──────────────────────────
    ax = axes[1, 0]
    ax.set_title("Element Stiffness  kₑ = E·Aₑ/h", fontsize=11)
    elem_nums = np.arange(1, n + 1)
    bars = ax.bar(elem_nums, k, color=C_FEA, edgecolor='white', alpha=0.85)
    ax.set_xlabel("Element number"); ax.set_ylabel("kₑ  (N/mm)")
    ax.set_xticks(elem_nums)
    for bar, kv in zip(bars, k):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{kv:.0f}', ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # ─── Plot 4: Convergence study ──────────────────────────
    ax = axes[1, 1]
    ax.set_title("Convergence: FEA tip δ vs exact  (varying n)", fontsize=11)
    ns     = range(1, 41)
    deltas = []
    for ni in ns:
        hi = L / ni
        ki = [E * (np.pi/4*(d1+(d2-d1)*(ei+0.5)/ni)**2) / hi for ei in range(ni)]
        deltas.append(sum(P/kk for kk in ki))
    delta_ex = exact_solution(P, L, E, d1, d2)
    errors = [abs(d - delta_ex)/delta_ex*100 for d in deltas]
    ax.semilogy(list(ns), errors, color=C_FEA, marker='.', linewidth=1.5, markersize=5)
    ax.axvline(n, color=C_EXACT, linewidth=1.5, linestyle='--', label=f'Current n={n}')
    ax.set_xlabel("Number of elements n"); ax.set_ylabel("Error (%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    # Save in the same folder as this script (works on Windows & Linux)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "tapered_bar_fem_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✔  Plot saved → {out_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FEM — Circular Tapered Bar   (Direct Stiffness Method) ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── INPUT ────────────────────────────────────────────────
    L, d1, d2, E, P, n = get_inputs()

    # ── STEP 1: Element properties ───────────────────────────
    h, x_nodes, d_nodes, x_mid, d_mid, A_mid, k = \
        compute_element_properties(n, L, d1, d2, E)
    print_element_table(n, h, x_mid, d_mid, A_mid, k)

    # ── STEP 2: Global K ─────────────────────────────────────
    K = assemble_global_K(n, k)
    print_global_K(K, n)

    # ── STEP 3: Apply BC & solve ─────────────────────────────
    print_reduced_system(K, P, n)
    u, f_full = apply_bc_and_solve(K, P, n)

    # ── STEP 4: Stresses & results ───────────────────────────
    stress, force = compute_element_stress(n, u, k, A_mid, E, h)
    print_results(n, u, x_nodes, stress, force, A_mid, P, L, E, d1, d2)

    # ── STEP 5: Plots ────────────────────────────────────────
    print()
    do_plot = input("  Generate plots? (y/n) : ").strip().lower()
    if do_plot == 'y':
        plot_results(n, L, d1, d2, x_nodes, u, k, A_mid, E, h, P)

    separator("═")
    print("  Done. ✔")
    separator("═")


if __name__ == "__main__":
    main()
