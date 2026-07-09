import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def _make_grid(n_items, max_cols=4, base_w=5, base_h=4):
    cols = min(max_cols, n_items)
    rows = math.ceil(n_items / max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(base_w * cols, base_h * rows))
    if n_items == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)
    return fig, axes


def _get_tsne_perplexity(n_samples: int) -> int:
    if n_samples <= 5:
        return 2
    return max(5, min(30, (n_samples - 1) // 3))


def _plot_tsne(ax, features, labels, class_names, model_name):
    n_samples = features.shape[0]
    perplexity = _get_tsne_perplexity(n_samples)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(features)

    labels = np.asarray(labels)
    colors = sns.color_palette("tab10", len(class_names))

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if np.any(mask):
            ax.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                s=18,
                alpha=0.72,
                color=colors[class_idx],
                edgecolors="white",
                linewidths=0.25,
                label=class_name,
            )

            centroid = tsne_results[mask].mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                s=140,
                color=colors[class_idx],
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            ax.text(
                centroid[0],
                centroid[1],
                f" {class_name}",
                fontsize=8,
                weight="bold",
                va="center",
                ha="left",
            )

    ax.set_title(f"{model_name}", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, which="major", linestyle="--", alpha=0.28)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.12)
    ax.legend(loc="best", fontsize=8, frameon=True)


def evaluate_and_plot_all(models_dict, test_loader, device, num_classes=4, output_dir=None, class_names=None):
    sns.set_theme(style="whitegrid", context="paper", palette="muted")

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    num_models = len(models_dict)
    test_accuracies = {}
    condition_accuracies = {}
    metrics_summary = []

    fig_roc, axes_roc = _make_grid(num_models, max_cols=4, base_w=5, base_h=4)
    fig_tsne, axes_tsne = _make_grid(num_models, max_cols=4, base_w=5, base_h=4)
    fig_cm, axes_cm = _make_grid(num_models, max_cols=4, base_w=5, base_h=4)

    print("=" * 60)
    print("🔬 QUANTITATIVE EXPERIMENT REPORT")
    print("=" * 60)

    for idx, (model_name, model) in enumerate(models_dict.items()):
        model.eval()
        features, all_preds, all_labels, all_probs = [], [], [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)

                try:
                    out = model(x, return_features=True)
                except TypeError:
                    out = model(x)

                if isinstance(out, (tuple, list)):
                    logits = out[0]
                    feats = out[1] if len(out) > 1 else logits
                else:
                    logits = out
                    feats = logits

                probs = torch.softmax(logits, dim=1)

                feats_np = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
                features.append(feats_np)
                all_probs.append(probs.detach().cpu().numpy())
                all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())

        features = np.concatenate(features, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        acc = accuracy_score(all_labels, all_preds)
        macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        test_accuracies[model_name] = acc
        metrics_summary.append(
            {
                "Model": model_name,
                "Accuracy": acc,
                "Precision": macro_p,
                "Recall": macro_r,
                "F1": macro_f1,
            }
        )

        print(f"\n🚀 Target: {model_name} | Test Accuracy: {acc:.4f}")
        print(
            classification_report(
                all_labels,
                all_preds,
                digits=4,
                target_names=class_names,
                zero_division=0,
            )
        )

        labels_bin = label_binarize(all_labels, classes=range(num_classes))
        ax_r = axes_roc[idx]
        roc_colors = sns.color_palette("husl", num_classes)
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            ax_r.plot(fpr, tpr, color=roc_colors[i], lw=2, label=f"{class_names[i]} (AUC={auc(fpr, tpr):.2f})")
        ax_r.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
        ax_r.set_title(f"{model_name} - ROC", fontweight="bold")
        ax_r.set_xlabel("False Positive Rate")
        ax_r.set_ylabel("True Positive Rate")
        ax_r.legend(loc="lower right", fontsize=8, frameon=True)
        ax_r.grid(True, linestyle="--", alpha=0.25)

        ax_t = axes_tsne[idx]
        _plot_tsne(ax_t, features, all_labels, class_names, model_name)

        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes), normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            ax=axes_cm[idx],
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"fontsize": 8},
        )
        axes_cm[idx].set_title(f"{model_name}", fontweight="bold")
        axes_cm[idx].set_xlabel("Predicted Label", fontsize=10)
        axes_cm[idx].set_ylabel("True Label", fontsize=10)

        class_acc = cm.diagonal()
        condition_accuracies[model_name] = class_acc

    for ax in axes_roc[num_models:]:
        ax.axis("off")
    for ax in axes_tsne[num_models:]:
        ax.axis("off")
    for ax in axes_cm[num_models:]:
        ax.axis("off")

    fig_acc, ax_acc = plt.subplots(figsize=(max(6, num_models * 1.5), 5))
    names = list(test_accuracies.keys())
    vals = [test_accuracies[n] for n in names]
    sns.barplot(x=names, y=vals, ax=ax_acc, palette="viridis")
    ax_acc.set_title("Overall Test Accuracy Across Models", fontweight="bold", pad=15)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.tick_params(axis="x", rotation=30)
    ax_acc.grid(axis="y", linestyle="--", alpha=0.25)

    for i, v in enumerate(vals):
        ax_acc.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig_cond, ax_cond = plt.subplots(figsize=(max(8, num_models * 2.5), 6))
    cond_data = []
    for m_name, accs in condition_accuracies.items():
        for c_idx, a in enumerate(accs):
            cond_data.append({"Model": m_name, "Condition": class_names[c_idx], "Accuracy": a})

    df_cond = pd.DataFrame(cond_data)
    sns.barplot(data=df_cond, x="Condition", y="Accuracy", hue="Model", ax=ax_cond, palette="Set2")
    ax_cond.set_title("Condition vs. Accuracy Across All Models", fontweight="bold", fontsize=14, pad=15)
    ax_cond.set_ylabel("Class Accuracy")
    ax_cond.set_xlabel("Condition (Class)")
    ax_cond.set_ylim(0, 1.05)
    ax_cond.legend(title="Models", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax_cond.grid(axis="y", linestyle="--", alpha=0.25)

    df_metrics = pd.DataFrame(metrics_summary).set_index("Model")
    fig_met, ax_met = plt.subplots(figsize=(max(8, num_models * 1.7), 5))
    sns.heatmap(
        df_metrics[["Accuracy", "Precision", "Recall", "F1"]],
        annot=True,
        fmt=".3f",
        cmap="crest",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Score"},
        ax=ax_met,
    )
    ax_met.set_title("Classification Metrics Summary", fontweight="bold", pad=12)
    ax_met.set_xlabel("Metric")
    ax_met.set_ylabel("Model")

    fig_roc.tight_layout()
    fig_tsne.tight_layout()
    fig_cm.tight_layout()
    fig_acc.tight_layout()
    fig_cond.tight_layout()
    fig_met.tight_layout()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_roc.savefig(out / "roc_curves.png", dpi=1200, bbox_inches="tight")
        fig_tsne.savefig(out / "tsne.png", dpi=1200, bbox_inches="tight")
        fig_cm.savefig(out / "normalized_confusion_matrices.png", dpi=1200, bbox_inches="tight")
        fig_acc.savefig(out / "accuracy_bar_chart.png", dpi=1200, bbox_inches="tight")
        fig_cond.savefig(out / "condition_accuracy_bar_chart.png", dpi=1200, bbox_inches="tight")
        fig_met.savefig(out / "classification_metrics_heatmap.png", dpi=1200, bbox_inches="tight")

        fig_roc.savefig(out / "roc_curves_low_res.png", dpi=300, bbox_inches="tight")
        fig_tsne.savefig(out / "tsne.png_low_res.png", dpi=300, bbox_inches="tight")
        fig_cm.savefig(out / "normalized_confusion_matrices_low_res.png", dpi=300, bbox_inches="tight")
        fig_acc.savefig(out / "accuracy_bar_chart_low_res.png", dpi=300, bbox_inches="tight")
        fig_cond.savefig(out / "condition_accuracy_bar_chart_low_res.png", dpi=300, bbox_inches="tight")
        fig_met.savefig(out / "classification_metrics_heatmap_low_res.png", dpi=300, bbox_inches="tight")
        print(f"Saved figures to: {out}")

    plt.show()