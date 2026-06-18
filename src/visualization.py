import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


METHOD_COLORS = {
    "Pearson": "#1f77b4",
    "Spearman": "#ff7f0e",
    "F-test": "#2ca02c",
    "Mutual Info": "#d62728",
    "RFE": "#9467bd",
    "SFS (Fwd)": "#8c564b",
    "SBS (Bwd)": "#e377c2",
    "Lasso (L1)": "#7f7f7f",
    "Random Forest": "#bcbd22",
}

METHOD_MARKERS = {
    "Pearson": "o",
    "Spearman": "v",
    "F-test": "^",
    "Mutual Info": ">",
    "RFE": "s",
    "SFS (Fwd)": "D",
    "SBS (Bwd)": "*",
    "Lasso (L1)": "d",
    "Random Forest": "p",
}

TABLE_HEADER_BG = "#1a3a5c"
TABLE_RANK_BG = "#e8e8e8"
TABLE_ROW_ALT = ["#f0f4f8", "#fafbfc"]


def plot_feature_selection_allinone(results, rankings, sweet_k=3, output_path="reports/figures/feature_selection_performance_allinone.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(20, 11), dpi=150)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2.1, 1.6], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.25,
                          left=0.06, right=0.98, top=0.94, bottom=0.05)

    ax_r2 = fig.add_subplot(gs[0, 0])
    ax_mse = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    method_names = [m for m in results.keys() if m != "Best (Frontier)"]

    for method in method_names:
        ks = results[method]["k"]
        r2_vals = results[method]["r2"]
        mse_vals = results[method]["mse"]
        color = METHOD_COLORS.get(method, "#333333")
        marker = METHOD_MARKERS.get(method, "o")

        ax_r2.plot(ks, r2_vals, marker=marker, color=color, linewidth=1.5,
                   markersize=7, label=method, alpha=0.85)
        ax_mse.plot(ks, mse_vals, marker=marker, color=color, linewidth=1.5,
                    markersize=7, label=method, alpha=0.85)

    ks_all = results[method_names[0]]["k"]
    best_r2 = [max(results[m]["r2"][i] for m in method_names) for i in range(len(ks_all))]
    best_mse = [min(results[m]["mse"][i] for m in method_names) for i in range(len(ks_all))]

    ax_r2.plot(ks_all, best_r2, marker="D", color="black", linewidth=2.5,
               markersize=8, label="Best (Frontier)", zorder=10)
    ax_mse.plot(ks_all, best_mse, marker="D", color="black", linewidth=2.5,
                markersize=8, label="Best (Frontier)", zorder=10)

    sweet_r2 = best_r2[sweet_k - 1]
    sweet_mse_val = best_mse[sweet_k - 1]
    ax_r2.axvline(x=sweet_k, color="red", linestyle=":", linewidth=2, alpha=0.8)
    ax_mse.axvline(x=sweet_k, color="red", linestyle=":", linewidth=2, alpha=0.8)
    ax_r2.annotate(f"Sweet Spot\n(k={sweet_k})", xy=(sweet_k, sweet_r2),
                   xytext=(sweet_k + 1.5, sweet_r2 - 0.02),
                   fontsize=10, fontweight="bold", color="red",
                   arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))
    ax_mse.annotate(f"Sweet Spot\n(k={sweet_k})", xy=(sweet_k, sweet_mse_val),
                    xytext=(sweet_k + 1.5, sweet_mse_val + 2),
                    fontsize=10, fontweight="bold", color="red",
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))

    ax_r2.set_xlabel("Number of Features in Model", fontsize=12, fontweight="bold")
    ax_r2.set_ylabel("Test R-squared (R²)", fontsize=12, fontweight="bold")
    ax_r2.set_title("Test R-squared Score by Feature Subset Size", fontsize=13, fontweight="bold")
    ax_r2.set_xticks(range(1, 14))
    ax_r2.grid(True, alpha=0.3)
    ax_r2.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

    ax_mse.set_xlabel("Number of Features in Model", fontsize=12, fontweight="bold")
    ax_mse.set_ylabel("Test Mean Squared Error (MSE)", fontsize=12, fontweight="bold")
    ax_mse.set_title("Test MSE by Feature Subset Size", fontsize=13, fontweight="bold")
    ax_mse.set_xticks(range(1, 14))
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    _draw_feature_ranking_table(ax_table, rankings, method_names)

    fig.suptitle(
        "CRISP-DM Step 4: 9 Feature Selection Algorithms Stepwise Evaluation (Boston Housing)",
        fontsize=18,
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


def _draw_feature_ranking_table(ax, rankings, method_names):
    ax.axis("off")

    col_labels = ["Rank"] + method_names
    n_cols = len(col_labels)
    n_rows = 13

    cell_text = []
    for rank in range(1, n_rows + 1):
        row = [str(rank)]
        for method in method_names:
            feat = rankings[method][rank - 1]
            row.append(feat)
        cell_text.append(row)

    cell_colors = []
    for r in range(n_rows):
        row_colors = []
        for c in range(n_cols):
            if c == 0:
                row_colors.append(TABLE_RANK_BG)
            else:
                row_colors.append(TABLE_ROW_ALT[r % 2])
        cell_colors.append(row_colors)

    col_colors = [TABLE_HEADER_BG] * n_cols

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=col_colors,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#b0c4de")
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif col == 0:
            cell.set_text_props(fontweight="bold", fontsize=10)
        else:
            cell.set_text_props(fontsize=9)
        cell.set_linewidth(0.5)

    ax.set_title("Feature Ranking by 9 Selection Algorithms", fontsize=13, fontweight="bold",
                  pad=10)
