"""Figures for the Recurrent Small LMs technical report repo."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = "/home/hatch/workspace/papers/report-repo/figures"
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ---------------------------------------------------------------- sweep curve
loops = np.array([1, 2, 4, 8, 12, 16])
ppl = np.array([29.38, 11.18, 4.41, 4.90, 4.91, 4.91])
ppl_lo = np.array([28.30, 10.84, 4.28, 4.75, 4.76, 4.76])
ppl_hi = np.array([30.52, 11.54, 4.55, 5.06, 5.07, 5.07])

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(loops, ppl_lo, ppl_hi, alpha=0.25, color="#B08D3E")
ax.plot(loops, ppl, "o-", color="#23201A", lw=2.2, ms=7, label="Perplexity")
ax.errorbar(loops, ppl, yerr=[ppl - ppl_lo, ppl_hi - ppl],
             fmt="none", ecolor="#8A6D2F", capsize=4, lw=1.4)
ax.axvline(4, color="#B08D3E", ls="--", lw=1.4, label="Trained depth (4 loops)")
ax.set_xlabel("Test-time loop count")
ax.set_ylabel("Perplexity (256 held-out sequences)")
ax.set_title("Loop-depth extrapolation saturates at the trained depth",
             fontweight="bold")
ax.set_xticks(loops)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_sweep_curve.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- architecture diagram
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
GOLD, INK, LIGHT = "#B08D3E", "#23201A", "#EBDEC0"

def block(x, y, w, h, label, sub="", fc=LIGHT, ec=GOLD):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.6)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2 + 0.12, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.28, sub, ha="center", va="center",
                fontsize=8, color=INK)

block(0.2, 1.2, 1.6, 1.6, "Prelude", "6 layers", fc="#F5EFDD")
# loop block with experts
loop = mpatches.FancyBboxPatch((2.4, 0.6), 3.4, 2.8,
                               boxstyle="round,pad=0.1", fc="white",
                               ec=INK, lw=2)
ax.add_patch(loop)
ax.text(4.1, 3.15, "Loop block  × 4", ha="center", fontsize=10,
        fontweight="bold", color=INK)
for i, (ex, ey) in enumerate([(2.8, 1.9), (3.7, 1.9), (4.6, 1.9),
                              (2.8, 1.0), (3.7, 1.0), (4.6, 1.0)]):
    lbl = f"E{i+1}" if i < 6 else "…"
    b = mpatches.FancyBboxPatch((ex, ey), 0.7, 0.6,
                               boxstyle="round,pad=0.03", fc=LIGHT,
                               ec=GOLD, lw=1.2)
    ax.add_patch(b)
    ax.text(ex + 0.35, ey + 0.3, lbl, ha="center", va="center", fontsize=8)
ax.text(4.1, 0.78, "8 experts / MoE + GQA", ha="center", fontsize=8,
        style="italic", color=INK)
block(6.4, 1.2, 1.6, 1.6, "Coda", "2 layers", fc="#F5EFDD")
ax.text(8.6, 2.0, "logits", ha="center", fontsize=10, color=INK,
        fontweight="bold")
for x0, x1 in [(1.8, 2.4), (5.8, 6.4), (8.0, 8.45)]:
    ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
ax.text(5.0, 0.25, "shared weights · iterated 4× at train time",
        ha="center", fontsize=8, style="italic", color=GOLD)
ax.set_title("OpenMythos-style 44M reference architecture",
             fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_architecture.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ KV-cache heatmap
batches = np.array([1, 2, 4, 8, 16, 32])
ctx = np.array([512, 1024, 2048, 4096])
# MHA cache MB, decimal: 2 * B * T * U * d * 2bytes / 1e6, U=12, d=512
U, d = 12, 512
cache = 2 * batches[:, None] * ctx[None, :] * U * d * 2 / 1e6

fig, ax = plt.subplots(figsize=(8, 4.6))
im = ax.imshow(cache, cmap="YlOrBr", aspect="auto", norm=matplotlib.colors.LogNorm())
ax.set_xticks(range(len(ctx))); ax.set_xticklabels(ctx)
ax.set_yticks(range(len(batches))); ax.set_yticklabels(batches)
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Batch size")
for i in range(len(batches)):
    for j in range(len(ctx)):
        v = cache[i, j]
        txt = f"{v:.0f}M" if v < 1000 else f"{v/1000:.1f}G"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                color="white" if v > 300 else INK, fontweight="bold")
fig.colorbar(im, ax=ax, label="MHA KV-cache (MB, decimal, fp16)")
ax.set_title("KV-cache explodes with batch × context — GQA cuts it 4×",
             fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_kv_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- 3D saturation surface plot
fig = plt.figure(figsize=(9, 5.5))
ax = fig.add_subplot(111, projection="3d")
xs = np.array([1, 2, 4, 8, 12, 16])
ys = np.array([0, 1])  # 0 = loss, 1 = perplexity (scaled)
loss = np.array([3.380, 2.414, 1.484, 1.590, 1.591, 1.591])
# normalise both to [0,1] for the surface
ln = (loss - loss.min()) / (loss.max() - loss.min())
pn = (ppl - ppl.min()) / (ppl.max() - ppl.min())
X, Y = np.meshgrid(xs, ys)
Z = np.vstack([ln, pn])
ax.plot_surface(X, Y, Z, cmap="YlOrBr", alpha=0.85, edgecolor="#8A6D2F",
                linewidth=0.4)
ax.scatter(xs, np.zeros_like(xs), ln, color=INK, s=28, depthshade=False)
ax.scatter(xs, np.ones_like(xs), pn, color="#B08D3E", s=28, depthshade=False)
ax.set_xlabel("Loops"); ax.set_ylabel("Metric")
ax.set_yticks([0, 1]); ax.set_yticklabels(["Loss", "Perplexity"])
ax.set_zlabel("Normalised value")
ax.set_title("Both metrics cliff-dive to loop 4, then flatline",
             fontweight="bold", pad=12)
ax.view_init(elev=22, azim=-58)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_3d_saturation.png", bbox_inches="tight")
plt.close(fig)

print("figures done")
