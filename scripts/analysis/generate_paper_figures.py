#!/usr/bin/env python3
"""
Generate all publication-quality figures for the manuscript.
Saves PDF files directly to Latex_paperwriting/.
"""

import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from collections import defaultdict
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.visualize.plot import plot_atoms

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
BASE    = PROJECT
RESULTS = PROJECT / "results"
FIGS    = PROJECT / "figures"

# ── Style ─────────────────────────────────────────────────────────────────────
COLW   = 3.46   # single column width (inches)
FULLW  = 7.09   # full page width (inches)

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "legend.fontsize":  7,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "axes.linewidth":   0.6,
    "xtick.major.width":0.5,
    "ytick.major.width":0.5,
    "lines.linewidth":  1.2,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

MODELS  = ["sevennet", "mattersim", "upet", "mace"]
MLABELS = {"sevennet":"SevenNet-0", "mattersim":"MatterSim",
           "upet":"UPET", "mace":"MACE-MP-0"}
MC      = {"sevennet":"#2196F3", "mattersim":"#4CAF50",
           "upet":"#FF9800", "mace":"#E91E63"}
MLS     = {"sevennet":"-", "mattersim":"--", "upet":"-.", "mace":":"}
MMRK    = {"sevennet":"o","mattersim":"s","upet":"^","mace":"D"}

STRUCTS = ["Mg2SiO4_balanced","Fe2SiO4_balanced",
           "TiFeO3_balanced","CaAl2Si2O8_balanced"]
SLBL    = {"Mg2SiO4_balanced":  r"Mg$_2$SiO$_4$",
           "Fe2SiO4_balanced":  r"Fe$_2$SiO$_4$",
           "TiFeO3_balanced":   r"TiFeO$_3$",
           "CaAl2Si2O8_balanced":r"CaAl$_2$Si$_2$O$_8$"}
SNAME   = {"Mg2SiO4_balanced":"Forsterite","Fe2SiO4_balanced":"Fayalite",
           "TiFeO3_balanced":"Ilmenite","CaAl2Si2O8_balanced":"Anorthite"}
NATOMS  = {"Mg2SiO4_balanced":336,"Fe2SiO4_balanced":336,
           "TiFeO3_balanced":360,"CaAl2Si2O8_balanced":208}

ELEMENT_COLORS = {
    "Mg":"#00AA44","Fe":"#CC4400","Ti":"#9900CC",
    "Ca":"#0066CC","Al":"#FF8800","Si":"#AA8800",
    "O": "#DD4444",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_md_log(model, struct):
    p = RESULTS / model / struct / "md.log"
    d = defaultdict(list)
    with open(p) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("Time"): continue
            parts = s.split()
            if len(parts)==5:
                try:
                    d["time"].append(float(parts[0]))
                    d["T"].append(float(parts[4]))
                    d["Epot"].append(float(parts[2]))
                except ValueError: pass
    return {k:np.array(v) for k,v in d.items()}

def get_bond_distances(model, struct, sp_a, sp_b, cutoff, stride=5):
    traj = Trajectory(str(RESULTS/model/struct/"md.traj"))
    dists = []
    for i in range(0, len(traj), stride):
        atoms = traj[i]
        pos  = atoms.get_positions()
        syms = np.array(atoms.get_chemical_symbols())
        cell = atoms.get_cell()
        inv  = np.linalg.inv(cell.T)
        ia   = np.where(syms==sp_a)[0]
        ib   = np.where(syms==sp_b)[0]
        if len(ia)==0 or len(ib)==0: return np.array([])
        for pa in pos[ia]:
            dv = pos[ib]-pa; df=dv@inv.T; df-=np.round(df)
            dr = df@cell.T; d=np.sqrt((dr**2).sum(1))
            if sp_a==sp_b: d=d[d>1e-3]
            dists.append(d[d<cutoff])
    return np.concatenate(dists) if dists else np.array([])

def get_angles(model, struct, e1, cen, e2, c1, c2, stride=5):
    traj = Trajectory(str(RESULTS/model/struct/"md.traj"))
    angs = []
    same = (e1==e2)
    for i in range(0, len(traj), stride):
        atoms = traj[i]
        pos  = atoms.get_positions()
        syms = np.array(atoms.get_chemical_symbols())
        cell = atoms.get_cell(); inv=np.linalg.inv(cell.T)
        ic   = np.where(syms==cen)[0]
        i1   = np.where(syms==e1)[0]
        i2   = np.where(syms==e2)[0]
        if len(ic)==0 or len(i1)==0 or len(i2)==0: return np.array([])
        def nbrs(pc,idx,cut):
            dv=pos[idx]-pc; df=dv@inv.T; df-=np.round(df)
            dr=df@cell.T; d=np.sqrt((dr**2).sum(1))
            m=(d>1e-3)&(d<cut); return dr[m], d[m]
        for ic_ in ic:
            pc=pos[ic_]
            v1,d1=nbrs(pc,i1,c1); v2,d2=nbrs(pc,i2,c2)
            if len(v1)==0 or len(v2)==0: continue
            n1=v1/d1[:,None]; n2=v2/d2[:,None]
            if same:
                dots=n1@n1.T; u,v=np.triu_indices(len(n1),k=1)
                cos=np.clip(dots[u,v],-1,1)
            else:
                cos=np.clip((n1@n2.T).ravel(),-1,1)
            angs.append(np.degrees(np.arccos(cos)))
    return np.concatenate(angs) if angs else np.array([])

def compute_gr(model, struct, sp_a, sp_b, rmax=6.0, nbins=200, stride=5):
    traj  = Trajectory(str(RESULTS/model/struct/"md.traj"))
    start = len(traj)//2
    bins  = np.linspace(0,rmax,nbins+1)
    bc    = 0.5*(bins[:-1]+bins[1:])
    hist  = np.zeros(nbins)
    vols,nas,nbs,nf = 0.,0.,0.,0
    for i in range(start,len(traj),stride):
        atoms=traj[i]; pos=atoms.get_positions()
        syms=np.array(atoms.get_chemical_symbols())
        cell=atoms.get_cell()
        if np.linalg.det(cell)<1.: continue
        inv=np.linalg.inv(cell.T)
        ia=np.where(syms==sp_a)[0]; ib=np.where(syms==sp_b)[0]
        if len(ia)==0 or len(ib)==0: return None,None
        vols+=atoms.get_volume(); nas+=len(ia); nbs+=len(ib); nf+=1
        for pa in pos[ia]:
            dv=pos[ib]-pa; df=dv@inv.T; df-=np.round(df)
            dr=df@cell.T; d=np.sqrt((dr**2).sum(1))
            if sp_a==sp_b: d=d[d>1e-3]
            h,_=np.histogram(d[d<rmax],bins=bins); hist+=h
    if nf==0: return None,None
    vol=vols/nf; na=nas/nf; nb=nbs/nf
    sv=(4/3)*np.pi*(bins[1:]**3-bins[:-1]**3)
    norm=na*nf*(nb/vol)*sv
    if sp_a==sp_b: norm*=0.5
    return bc, hist/norm

def save(fig, name):
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Crystal structures (2×2 panel)
# ══════════════════════════════════════════════════════════════════════════════
def fig_structures():
    print("Fig 1: crystal structures...")
    fig, axes = plt.subplots(2, 2, figsize=(FULLW, FULLW*0.75))
    rotations = [
        ("Mg2SiO4_balanced",    "-90x,5y,0z"),
        ("Fe2SiO4_balanced",    "-90x,5y,0z"),
        ("TiFeO3_balanced",     "-60x,10y,0z"),
        ("CaAl2Si2O8_balanced", "-70x,15y,0z"),
    ]
    for ax, (struct, rot) in zip(axes.flatten(), rotations):
        atoms = ase_read(str(BASE/"structures"/f"{struct}.xyz"))
        # use a single unit cell for clarity: take first formula unit
        # limit to ~50 atoms for visual clarity
        plot_atoms(atoms, ax, rotation=rot, radii=0.35, show_unit_cell=1)
        ax.set_title(f"{SLBL[struct]}\n({SNAME[struct]}, {NATOMS[struct]} atoms)",
                     pad=3, fontsize=8)
        ax.set_axis_off()

    # element legend
    elements = {"Mg":"#00AA44","Fe":"#CC4400","Ti":"#9900CC",
                "Ca":"#0066CC","Al":"#FF8800","Si":"#AA8800","O":"#DD4444"}
    handles = [mpatches.Patch(color=c, label=e) for e,c in elements.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Crystal structures of the four lunar regolith minerals",
                 fontsize=9, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig1_structures.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Temperature & energy evolution
# ══════════════════════════════════════════════════════════════════════════════
def fig_temperature():
    print("Fig 2: temperature evolution...")
    fig, axes = plt.subplots(2, 4, figsize=(FULLW, FULLW*0.55))

    for col, struct in enumerate(STRUCTS):
        info = json.loads((RESULTS/"sevennet"/struct/"simulation_details.json").read_text())
        natoms = info["num_atoms"]
        ax_T = axes[0, col]
        ax_E = axes[1, col]
        for m in MODELS:
            d = load_md_log(m, struct)
            n = json.loads((RESULTS/m/struct/"simulation_details.json").read_text())["num_atoms"]
            t_fs = d["time"]*1000
            ax_T.plot(t_fs, d["T"], color=MC[m], ls=MLS[m], lw=0.9,
                      label=MLABELS[m], alpha=0.9)
            ax_E.plot(t_fs, d["Epot"]/n, color=MC[m], ls=MLS[m],
                      lw=0.9, alpha=0.9)
        ax_T.axhline(300, color="0.5", ls=":", lw=0.7, label="300 K")
        ax_T.set_title(f"{SLBL[struct]}", fontsize=8)
        ax_T.set_xlim(0,1000)
        ax_E.set_xlim(0,1000)
        ax_E.set_xlabel("Time (fs)", fontsize=7)
        if col == 0:
            ax_T.set_ylabel("Temperature (K)", fontsize=7)
            ax_E.set_ylabel(r"$E_\mathrm{pot}$/atom (eV)", fontsize=7)
        else:
            ax_T.set_yticklabels([]); ax_E.set_yticklabels([])
        ax_T.grid(alpha=0.2, lw=0.4); ax_E.grid(alpha=0.2, lw=0.4)
        ax_T.tick_params(labelbottom=False)

    # shared legend
    handles = [Line2D([],[],color=MC[m],ls=MLS[m],lw=1.2,label=MLABELS[m])
               for m in MODELS]
    handles += [Line2D([],[],color="0.5",ls=":",lw=0.7,label="300 K target")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5,-0.01))
    fig.subplots_adjust(hspace=0.08, wspace=0.06, bottom=0.12)
    save(fig, "fig2_temperature.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Bond distance histograms (one panel per mineral)
# ══════════════════════════════════════════════════════════════════════════════
BOND_PAIRS = {
    "Mg2SiO4_balanced":    [("Si","O",2.0,(1.40,2.00)), ("Mg","O",2.7,(1.80,2.70))],
    "Fe2SiO4_balanced":    [("Si","O",2.0,(1.40,2.00)), ("Fe","O",2.7,(1.80,2.70))],
    "TiFeO3_balanced":     [("Ti","O",2.4,(1.60,2.40)), ("Fe","O",2.7,(1.80,2.70))],
    "CaAl2Si2O8_balanced": [("Si","O",2.0,(1.40,2.00)), ("Al","O",2.2,(1.50,2.20)),
                             ("Ca","O",3.0,(2.00,3.00))],
}
# Experimental reference values (Å)
EXP_BONDS = {
    ("Mg2SiO4_balanced","Si","O"):  1.644,
    ("Mg2SiO4_balanced","Mg","O"):  2.100,
    ("Fe2SiO4_balanced","Si","O"):  1.636,
    ("Fe2SiO4_balanced","Fe","O"):  2.130,
    ("TiFeO3_balanced","Ti","O"):   2.045,
    ("TiFeO3_balanced","Fe","O"):   2.125,
    ("CaAl2Si2O8_balanced","Si","O"):1.634,
    ("CaAl2Si2O8_balanced","Al","O"):1.735,
    ("CaAl2Si2O8_balanced","Ca","O"):2.500,
}

def fig_bond_distances():
    print("Fig 3: bond distance histograms (computing...)")
    # pre-compute
    bdata = {}
    for struct, pairs in BOND_PAIRS.items():
        for sp_a, sp_b, cut, _ in pairs:
            for m in MODELS:
                print(f"  {m:10s} {struct:25s} {sp_a}-{sp_b}", end="\r")
                bdata[(m,struct,sp_a,sp_b)] = get_bond_distances(
                    m, struct, sp_a, sp_b, cut, stride=5)
    print()

    n_cols = max(len(v) for v in BOND_PAIRS.values())
    fig, axes = plt.subplots(4, n_cols, figsize=(FULLW, FULLW*1.0),
                              gridspec_kw={"wspace":0.35,"hspace":0.55})

    for row, struct in enumerate(STRUCTS):
        pairs = BOND_PAIRS[struct]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(pairs):
                ax.set_visible(False); continue
            sp_a, sp_b, cut, xr = pairs[col]
            bins = np.linspace(xr[0], xr[1], 55)
            for m in MODELS:
                arr = bdata.get((m,struct,sp_a,sp_b), np.array([]))
                if len(arr)==0: continue
                cnt, edges = np.histogram(arr, bins=bins, density=True)
                cen = 0.5*(edges[:-1]+edges[1:])
                ax.plot(cen, cnt, color=MC[m], ls=MLS[m], lw=1.1,
                        label=MLABELS[m], alpha=0.9)
                ax.fill_between(cen, cnt, alpha=0.06, color=MC[m])
            exp = EXP_BONDS.get((struct,sp_a,sp_b))
            if exp:
                ax.axvline(exp, color="0.3", ls="--", lw=0.8, label=f"Exp. {exp:.3f}\u00c5")
            ax.set_xlabel(f"{sp_a}–{sp_b} (Å)", fontsize=7)
            if col==0: ax.set_ylabel("Prob. density", fontsize=7)
            ax.set_title(f"{SLBL[struct]}: {sp_a}–{sp_b}", fontsize=7, pad=2)
            ax.grid(alpha=0.2, lw=0.4)
            ax.set_xlim(xr)

    # legend in last visible axis
    handles = [Line2D([],[],color=MC[m],ls=MLS[m],lw=1.1,label=MLABELS[m])
               for m in MODELS]
    handles += [Line2D([],[],color="0.3",ls="--",lw=0.8,label="Exp. ref.")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5,-0.01))
    save(fig, "fig3_bond_distances.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Bond angle distributions
# ══════════════════════════════════════════════════════════════════════════════
ANGLE_TRIPLETS = {
    "Mg2SiO4_balanced":    [("O","Si","O",2.0,2.0, r"O–Si–O",(85,135)),
                             ("Mg","O","Mg",2.7,2.7,r"Mg–O–Mg",(85,135)),
                             ("Mg","O","Si",2.7,2.0,r"Mg–O–Si",(85,165))],
    "Fe2SiO4_balanced":    [("O","Si","O",2.0,2.0, r"O–Si–O",(85,135)),
                             ("Fe","O","Fe",2.7,2.7,r"Fe–O–Fe",(80,140)),
                             ("Fe","O","Si",2.7,2.0,r"Fe–O–Si",(80,160))],
    "TiFeO3_balanced":     [("O","Ti","O",2.4,2.4, r"O–Ti–O",(55,180)),
                             ("O","Fe","O",2.7,2.7, r"O–Fe–O",(55,180)),
                             ("Ti","O","Fe",2.4,2.7,r"Ti–O–Fe",(80,160))],
    "CaAl2Si2O8_balanced": [("O","Si","O",2.0,2.0, r"O–Si–O",(85,135)),
                             ("O","Al","O",2.2,2.2, r"O–Al–O",(85,135)),
                             ("Si","O","Al",2.0,2.2,r"Si–O–Al",(105,175))],
}

def fig_bond_angles():
    print("Fig 4: bond angle histograms (computing...)")
    adata = {}
    for struct, trips in ANGLE_TRIPLETS.items():
        for e1,cen,e2,c1,c2,lbl,xr in trips:
            for m in MODELS:
                print(f"  {m:10s} {struct:25s} {lbl}", end="\r")
                adata[(m,struct,lbl)] = get_angles(m,struct,e1,cen,e2,c1,c2,stride=5)
    print()

    fig, axes = plt.subplots(4, 3, figsize=(FULLW, FULLW*0.95),
                              gridspec_kw={"wspace":0.35,"hspace":0.55})
    for row, struct in enumerate(STRUCTS):
        for col,(e1,cen,e2,c1,c2,lbl,xr) in enumerate(ANGLE_TRIPLETS[struct]):
            ax = axes[row, col]
            bins = np.linspace(xr[0], xr[1], 50)
            for m in MODELS:
                arr = adata.get((m,struct,lbl), np.array([]))
                if len(arr)==0: continue
                cnt,edges = np.histogram(arr, bins=bins, density=True)
                cen2 = 0.5*(edges[:-1]+edges[1:])
                ax.plot(cen2, cnt, color=MC[m], ls=MLS[m], lw=1.1, alpha=0.9,
                        label=MLABELS[m])
                ax.fill_between(cen2, cnt, alpha=0.06, color=MC[m])
            # tetrahedral reference for O-X-O
            if e1=="O" and e2=="O":
                ax.axvline(109.47, color="0.35", ls="--", lw=0.8,
                           label=r"Tet. 109.47°")
            ax.set_xlabel(f"{lbl} (°)", fontsize=7)
            if col==0: ax.set_ylabel("Prob. density", fontsize=7)
            ax.set_title(f"{SLBL[struct]}: {lbl}", fontsize=7, pad=2)
            ax.grid(alpha=0.2, lw=0.4); ax.set_xlim(xr)

    handles = [Line2D([],[],color=MC[m],ls=MLS[m],lw=1.1,label=MLABELS[m])
               for m in MODELS]
    handles += [Line2D([],[],color="0.35",ls="--",lw=0.8,label=r"Tetrahedral 109.47°")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5,-0.01))
    save(fig, "fig4_bond_angles.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Radial distribution functions
# ══════════════════════════════════════════════════════════════════════════════
GR_PAIRS = {
    "Mg2SiO4_balanced":    [("Si","O"),("Mg","O"),("O","O")],
    "Fe2SiO4_balanced":    [("Si","O"),("Fe","O"),("O","O")],
    "TiFeO3_balanced":     [("Ti","O"),("Fe","O"),("O","O")],
    "CaAl2Si2O8_balanced": [("Si","O"),("Al","O"),("Ca","O"),("O","O")],
}

def fig_gr():
    print("Fig 5: g(r) (computing from last 50% of trajectory...)")
    gr_data = {}
    for struct, pairs in GR_PAIRS.items():
        for sp_a, sp_b in pairs:
            for m in MODELS:
                print(f"  {m:10s} {struct:25s} g({sp_a}-{sp_b})", end="\r")
                r, gr = compute_gr(m, struct, sp_a, sp_b, stride=5)
                gr_data[(m,struct,sp_a,sp_b)] = (r, gr)
    print()

    n_cols = max(len(v) for v in GR_PAIRS.values())
    fig, axes = plt.subplots(4, n_cols, figsize=(FULLW, FULLW*1.0),
                              gridspec_kw={"wspace":0.35,"hspace":0.55})

    for row, struct in enumerate(STRUCTS):
        pairs = GR_PAIRS[struct]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(pairs):
                ax.set_visible(False); continue
            sp_a, sp_b = pairs[col]
            for m in MODELS:
                r, gr = gr_data.get((m,struct,sp_a,sp_b),(None,None))
                if r is None: continue
                ax.plot(r, gr, color=MC[m], ls=MLS[m], lw=1.0,
                        label=MLABELS[m], alpha=0.9)
            ax.axhline(1.0, color="0.5", ls=":", lw=0.6)
            ax.set_xlabel("r (Å)", fontsize=7)
            if col==0: ax.set_ylabel("g(r)", fontsize=7)
            ax.set_title(f"{SLBL[struct]}: g({sp_a}–{sp_b})", fontsize=7, pad=2)
            ax.set_xlim(0.8, 6.0); ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2, lw=0.4)

    handles = [Line2D([],[],color=MC[m],ls=MLS[m],lw=1.1,label=MLABELS[m])
               for m in MODELS]
    handles += [Line2D([],[],color="0.5",ls=":",lw=0.7,label="g(r)=1")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5,-0.01))
    save(fig, "fig5_gr.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Computational performance (time + memory)
# ══════════════════════════════════════════════════════════════════════════════
def fig_performance():
    print("Fig 6: performance...")
    MEMDIR = RESULTS / "memory"
    mem = {m: json.loads((MEMDIR/f"{m}_memory.json").read_text()) for m in MODELS}
    timing = {m: {s: json.loads((RESULTS/m/s/"simulation_details.json").read_text())
                  for s in STRUCTS} for m in MODELS}

    GPU_TOT = mem["mace"]["gpu_info"]["total_mb"]
    params  = {"sevennet":842623,"mattersim":890034,"upet":108629786,"mace":9063204}
    dtype   = {"sevennet":"float32","mattersim":"float32","upet":"float32","mace":"float64"}

    fig = plt.figure(figsize=(FULLW, FULLW*0.72))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.40, hspace=0.52)

    short = [r"Mg$_2$SiO$_4$",r"Fe$_2$SiO$_4$",r"TiFeO$_3$",r"CaAl$_2$Si$_2$O$_8$"]
    x     = np.arange(len(STRUCTS))
    nmod  = len(MODELS)
    w     = 0.17

    # (a) Wall-clock time
    ax_a = fig.add_subplot(gs[0, :2])
    for j, m in enumerate(MODELS):
        times = [timing[m][s]["simulation_time_s"] for s in STRUCTS]
        offs  = x + j*w - (nmod-1)*w/2
        bars  = ax_a.bar(offs, times, w, label=MLABELS[m],
                         color=MC[m], edgecolor="w", lw=0.4)
    ax_a.set_xticks(x); ax_a.set_xticklabels(short, fontsize=7)
    ax_a.set_ylabel("Wall-clock time (s)\nfor 1000 MD steps", fontsize=7)
    ax_a.set_title("(a) Simulation wall-clock time", fontsize=8)
    ax_a.legend(fontsize=6.5, ncol=2, loc="upper right",
                framealpha=0.8, edgecolor="0.8")
    ax_a.grid(axis="y", alpha=0.25, lw=0.4)

    # (b) ms per step (time normalised)
    ax_b = fig.add_subplot(gs[0, 2])
    ms_per_step = {m: np.mean([timing[m][s]["simulation_time_s"]/
                                timing[m][s]["num_steps"]*1000
                                for s in STRUCTS]) for m in MODELS}
    bars_b = ax_b.barh([MLABELS[m] for m in MODELS],
                        [ms_per_step[m] for m in MODELS],
                        color=[MC[m] for m in MODELS], edgecolor="w", lw=0.4)
    for bar, m in zip(bars_b, MODELS):
        ax_b.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                  f"{ms_per_step[m]:.0f} ms", va="center", fontsize=6.5)
    ax_b.set_xlabel("Mean ms per step\n(avg. over 4 structures)", fontsize=7)
    ax_b.set_title("(b) Mean step time", fontsize=8)
    ax_b.grid(axis="x", alpha=0.25, lw=0.4)
    ax_b.set_xlim(0, max(ms_per_step.values())*1.30)

    # (c) Peak GPU memory
    ax_c = fig.add_subplot(gs[1, :2])
    for j, m in enumerate(MODELS):
        peaks = [mem[m]["structures"][s]["peak_gpu_during_run_mb"] for s in STRUCTS]
        offs  = x + j*w - (nmod-1)*w/2
        ax_c.bar(offs, peaks, w, label=MLABELS[m],
                 color=MC[m], edgecolor="w", lw=0.4)
    ax_c.axhline(GPU_TOT, color="0.4", ls="--", lw=0.7,
                 label=f"GPU capacity ({GPU_TOT/1024:.0f} GB)")
    ax_c.set_xticks(x); ax_c.set_xticklabels(short, fontsize=7)
    ax_c.set_ylabel("Peak GPU memory (MB)", fontsize=7)
    ax_c.set_title("(c) Peak GPU memory during MD", fontsize=8)
    ax_c.legend(fontsize=6.5, ncol=3, framealpha=0.8, edgecolor="0.8")
    ax_c.grid(axis="y", alpha=0.25, lw=0.4)

    # (d) Params vs mean step time scatter
    ax_d = fig.add_subplot(gs[1, 2])
    for m in MODELS:
        peak_rep = mem[m]["structures"]["TiFeO3_balanced"]["peak_gpu_during_run_mb"]
        ax_d.scatter(params[m], ms_per_step[m], color=MC[m],
                     s=60, marker=MMRK[m], zorder=5, label=MLABELS[m])
        ax_d.annotate(f"{MLABELS[m]}\n[{dtype[m]}]",
                      (params[m], ms_per_step[m]),
                      xytext=(5,3), textcoords="offset points",
                      fontsize=5.5, color=MC[m])
    ax_d.set_xscale("log")
    ax_d.set_xlabel("Parameters (log scale)", fontsize=7)
    ax_d.set_ylabel("Mean step time (ms)", fontsize=7)
    ax_d.set_title("(d) Size vs. speed", fontsize=8)
    ax_d.grid(alpha=0.2, lw=0.4)

    save(fig, "fig6_performance.pdf")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    # run in order; skip slow g(r)/bond if only quick test
    all_figs = True if (len(sys.argv)<2 or sys.argv[1]=="all") else False
    quick    = len(sys.argv)>1 and sys.argv[1]=="quick"

    fig_structures()
    fig_temperature()
    fig_performance()

    if not quick:
        fig_bond_distances()
        fig_bond_angles()
        fig_gr()

    print("\nAll figures saved to:", FIGS)
