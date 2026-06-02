import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_and_plot_all(models_dict, test_loader, device, num_classes=4):
    num_models = len(models_dict)
    test_accuracies = {}
    
    fig_roc, axes_roc = plt.subplots(nrows=1, ncols=num_models, figsize=(5 * num_models, 5))
    fig_tsne, axes_tsne = plt.subplots(nrows=1, ncols=num_models, figsize=(5 * num_models, 5))
    fig_cm, axes_cm = plt.subplots(nrows=1, ncols=num_models, figsize=(5 * num_models, 5))
    
    if num_models == 1:
        axes_roc, axes_tsne, axes_cm = [axes_roc], [axes_tsne], [axes_cm]

    print("="*60)
    print("🔬 QUANTITATIVE EXPERIMENT REPORT")
    print("="*60)

    for idx, (model_name, model) in enumerate(models_dict.items()):
        model.eval()
        features, all_preds, all_labels, all_probs = [], [], [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                logits, feats = model(x.to(device), return_features=True)
                probs = torch.softmax(logits, dim=1)
                
                features.append(feats.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(y.numpy())
                
        features = np.concatenate(features, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels, all_preds = np.array(all_labels), np.array(all_preds)
        
        acc = accuracy_score(all_labels, all_preds)
        test_accuracies[model_name] = acc
        
        print(f"\n🚀 Target: {model_name} | Test Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds, digits=4, target_names=[f"Class {i}" for i in range(num_classes)]))
        
        # Plotting ROC
        labels_bin = label_binarize(all_labels, classes=range(num_classes))
        ax_r = axes_roc[idx]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            ax_r.plot(fpr, tpr, label=f'C{i} (AUC = {auc(fpr, tpr):.2f})')
        ax_r.plot([0, 1], [0, 1], 'k--')
        ax_r.set_title(model_name)
        ax_r.legend(loc="lower right")

        # Plotting t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(features)
        ax_t = axes_tsne[idx]
        scatter = ax_t.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis', s=15)
        ax_t.set_title(model_name)
        
        # Confusion Matrices
        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], cbar=False)
        axes_cm[idx].set_title(model_name)

    plt.show()