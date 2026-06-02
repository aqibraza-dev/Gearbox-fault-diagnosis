import torch
from sklearn.metrics import accuracy_score

def evaluate_on_test(model, test_loader, device):
    """Calculates pure accuracy evaluations on validation tensors."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    return accuracy_score(all_labels, all_preds)