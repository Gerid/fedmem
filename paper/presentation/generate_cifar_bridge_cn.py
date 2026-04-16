from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


OUT_DIR = Path(r"E:\fedprotrack\paper\presentation\assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "cifar_bridge_cn.png"

FONT_REG = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = fm.FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")


def apply_font(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_REG)


def main():
    scenes = ["差别明显", "差别较小"]
    pooled = [71.83, 50.17]
    split = [85.90, 61.00]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    x = range(len(scenes))
    width = 0.32

    ax.bar([i - width / 2 for i in x], pooled, width=width, color="#7aa6c2", label="合在一起训练")
    ax.bar([i + width / 2 for i in x], split, width=width, color="#e39a43", label="按类别分别训练")

    for i, v in enumerate(pooled):
        ax.text(i - width / 2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontproperties=FONT_REG)
    for i, v in enumerate(split):
        ax.text(i + width / 2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontproperties=FONT_REG)

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenes)
    ax.set_ylim(0, 100)
    ax.set_ylabel("准确率（越高越好）", fontproperties=FONT_REG, fontsize=11)
    ax.set_title("真实图片数据结果：按类别分别训练更好", fontproperties=FONT_BOLD, fontsize=15, pad=12)
    apply_font(ax)

    legend = ax.legend(frameon=False, loc="upper left", prop=FONT_REG)
    for text in legend.get_texts():
        text.set_fontproperties(FONT_REG)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.text(
        0.02,
        -0.20,
        "根据论文中的 CIFAR-100 两组实验结果整理，数值为 3 次实验平均。",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        fontproperties=FONT_REG,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
