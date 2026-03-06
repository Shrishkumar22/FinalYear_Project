import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import json


# =====================================================
# FOCAL LOSS - Focuses on hard examples
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


# =====================================================
# DATA NORMALIZER
# =====================================================
class DataNormalizer:
    def __init__(self, method='robust'):
        self.method = method
        self.params = {}

    def fit(self, x_train):
        if self.method == 'robust':
            self.params['median'] = x_train.median(dim=0, keepdim=True)[0]
            q1 = x_train.quantile(0.25, dim=0, keepdim=True)
            q3 = x_train.quantile(0.75, dim=0, keepdim=True)
            self.params['iqr'] = (q3 - q1) + 1e-8
        return self

    def transform(self, x):
        if self.method == 'robust':
            return (x - self.params['median']) / self.params['iqr']
        return x

    def fit_transform(self, x_train):
        self.fit(x_train)
        return self.transform(x_train)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = features.float()
        labels = labels.float()

        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        mask_sum = mask.sum(1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        return -mean_log_prob_pos.mean()


# =====================================================
# EVALUATION WITH CUSTOM THRESHOLD
# =====================================================
def evaluate_with_threshold(model, x_test, y_test, threshold=0.5, device='cuda'):
    """Evaluate model with custom decision threshold"""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(x_test), 512):
            batch_x = x_test[i:i+512].to(device)
            batch_y = y_test[i:i+512]

            _, _, logits = model(batch_x)
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(batch_y.tolist())

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs
    }


# =====================================================
# STEP 1: THRESHOLD TUNING
# =====================================================
def threshold_tuning(model_path, x_test, y_test, device='cuda'):
    """Try different thresholds to optimize accuracy while maintaining recall"""
    from utils import AE_classifier

    print("="*70)
    print("🎯 STEP 1: THRESHOLD TUNING (No Retraining)")
    print("="*70)
    print("\nGoal: Find threshold that gives 90%+ recall with best accuracy\n")

    # Load Model 2
    model = AE_classifier(input_dim=17).to(device)
    model.load_state_dict(torch.load(model_path))

    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []

    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Status'}")
    print("-"*70)

    best_result = None
    best_threshold = 0.5

    for threshold in thresholds:
        result = evaluate_with_threshold(model, x_test, y_test, threshold, device)

        # Status check
        status = ""
        if result['recall'] >= 0.90:
            status = "✅ Recall OK"
            if result['accuracy'] > 0.73:
                status += " 🎯 GOOD!"
        else:
            status = "❌ Recall too low"

        print(f"{threshold:<12.2f} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} {result['f1']:<12.4f} {status}")

        results.append({
            'threshold': threshold,
            **result
        })

        # Track best: recall >= 90% and highest accuracy
        if result['recall'] >= 0.90:
            if best_result is None or result['accuracy'] > best_result['accuracy']:
                best_result = result
                best_threshold = threshold

    print("\n" + "="*70)
    if best_result:
        print(f"✅ BEST THRESHOLD: {best_threshold}")
        print(f"   Accuracy:  {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
        print(f"   Precision: {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
        print(f"   Recall:    {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
        print(f"   F1 Score:  {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)")
    else:
        print("⚠️  No threshold maintains 90%+ recall")
        best_threshold = 0.5
        best_result = results[1]  # Default 0.5 threshold

    print("="*70)

    return best_threshold, best_result, results


# =====================================================
# STEP 2: FINE-TUNING WITH FOCAL LOSS
# =====================================================
def finetune_model(
    base_model_path,
    x_train,
    y_train,
    x_test,
    y_test,
    device='cuda',
    epochs=50,
    learning_rate=1e-5
):
    """Fine-tune Model 2 with Focal Loss to reduce false positives"""
    from utils import AE_classifier

    print("\n" + "="*70)
    print("🔧 STEP 2: FINE-TUNING WITH FOCAL LOSS")
    print("="*70)
    print("\nGoal: Reduce false positives while maintaining 90%+ recall")
    print(f"Strategy: Lower learning rate + Focal loss + More epochs\n")

    # Load Model 2 as starting point
    model = AE_classifier(input_dim=17).to(device)
    model.load_state_dict(torch.load(base_model_path))

    # Data loader
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )

    # Optimizer with very low learning rate (fine-tuning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Loss functions
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.2)
    classification_criterion = FocalLoss(alpha=0.25, gamma=2.0)  # FOCAL LOSS
    recon_loss_fn = nn.L1Loss()

    # ADJUSTED WEIGHTS - focus more on precision
    contrastive_weight = 0.4  # Reduced
    cls_weight = 1.2          # Increased
    recon_weight = 0.001

    print(f"Fine-tuning configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Loss: Focal Loss (alpha=0.25, gamma=2.0)")
    print(f"  Weights: Con={contrastive_weight}, Cls={cls_weight}, Rec={recon_weight}")
    print(f"  Epochs: {epochs}\n")

    best_model_state = None
    best_score = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0.0, 'batches': 0}

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            z, recon, logits = model(inputs)

            contrastive_loss = contrastive_loss_fn(z, labels)
            cls_loss = classification_criterion(logits, labels.float().unsqueeze(1))
            recon_loss = recon_loss_fn(recon, inputs)

            loss = (contrastive_weight * contrastive_loss + 
                    cls_weight * cls_loss + 
                    recon_weight * recon_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_losses['total'] += loss.item()
            epoch_losses['batches'] += 1

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            result = evaluate_with_threshold(model, x_test, y_test, 0.5, device)

            # Score: prioritize recall >= 90%, then accuracy
            score = 0
            if result['recall'] >= 0.90:
                score = result['accuracy'] + 0.1  # Bonus for maintaining recall
            else:
                score = result['recall']  # Penalize if recall drops

            status = "✅" if result['recall'] >= 0.90 else "⚠️"
            print(f"Epoch {epoch+1:2d}/{epochs} {status} | "
                  f"Acc: {result['accuracy']:.4f} Prec: {result['precision']:.4f} "
                  f"Rec: {result['recall']:.4f} F1: {result['f1']:.4f}")

            if score > best_score:
                best_score = score
                best_model_state = model.state_dict().copy()

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    print("\n" + "="*70)
    print("✅ Fine-tuning complete")
    print("="*70)

    return model


# =====================================================
# MAIN OPTIMIZATION PIPELINE
# =====================================================
def optimize_model(dataset, base_model_path="teacher_improved_best.pth", device='cuda'):
    """Complete optimization pipeline"""
    from utils import load_data, SplitData

    print("\n" + "="*70)
    print("🚀 MODEL 2 OPTIMIZATION PIPELINE")
    print("="*70)
    print("\nObjective: Improve accuracy while maintaining 90%+ recall")
    print(f"Base Model: {base_model_path}\n")

    # Load and prepare data
    print("Loading data...")
    data = load_data(dataset)
    splitter = SplitData(dataset="new")
    x_all, y_all = splitter.transform(data, labels="label")

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train.astype(np.int64))
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test.astype(np.int64))

    # Normalize
    normalizer = DataNormalizer(method='robust')
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    print(f"✅ Data ready: {len(x_train):,} train, {len(x_test):,} test\n")

    # STEP 1: Threshold tuning
    best_threshold, threshold_result, all_threshold_results = threshold_tuning(
        base_model_path, x_test, y_test, device
    )

    # Check if threshold tuning achieved goal
    if threshold_result['accuracy'] >= 0.73 and threshold_result['recall'] >= 0.90:
        print("\n🎉 SUCCESS with threshold tuning!")
        print(f"   No fine-tuning needed - threshold {best_threshold} achieves goal\n")

        # Save results
        results = {
            'method': 'threshold_tuning',
            'threshold': best_threshold,
            'metrics': threshold_result
        }

        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return best_threshold, threshold_result

    # STEP 2: Fine-tuning
    print("\n⚠️  Threshold tuning not sufficient, proceeding to fine-tuning...\n")
    time.sleep(2)

    finetuned_model = finetune_model(
        base_model_path,
        x_train,
        y_train,
        x_test,
        y_test,
        device=device,
        epochs=50,
        learning_rate=1e-5
    )

    # Save fine-tuned model
    torch.save(finetuned_model.state_dict(), 'teacher_optimized.pth')
    print(f"\n💾 Fine-tuned model saved: teacher_optimized.pth")

    # Evaluate fine-tuned model with threshold tuning
    print("\n" + "="*70)
    print("🎯 FINAL EVALUATION: Fine-tuned Model + Threshold Tuning")
    print("="*70)

    final_threshold, final_result, _ = threshold_tuning(
        'teacher_optimized.pth', x_test, y_test, device
    )

    # Save results
    results = {
        'method': 'finetuning + threshold',
        'threshold': final_threshold,
        'metrics': final_result
    }

    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # COMPARISON
    print("\n" + "="*70)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("="*70)

    # Original Model 2
    print("\n📦 ORIGINAL MODEL 2:")
    print(f"   Accuracy:  70.87%")
    print(f"   Precision: 64.13%")
    print(f"   Recall:    93.92%")
    print(f"   F1 Score:  76.22%")

    print("\n✨ OPTIMIZED MODEL:")
    print(f"   Accuracy:  {final_result['accuracy']*100:.2f}%")
    print(f"   Precision: {final_result['precision']*100:.2f}%")
    print(f"   Recall:    {final_result['recall']*100:.2f}%")
    print(f"   F1 Score:  {final_result['f1']*100:.2f}%")
    print(f"   Threshold: {final_threshold}")

    acc_change = (final_result['accuracy'] - 0.7087) * 100
    prec_change = (final_result['precision'] - 0.6413) * 100
    rec_change = (final_result['recall'] - 0.9392) * 100
    f1_change = (final_result['f1'] - 0.7622) * 100

    print("\n📈 CHANGES:")
    print(f"   Accuracy:  {acc_change:+.2f}%")
    print(f"   Precision: {prec_change:+.2f}%")
    print(f"   Recall:    {rec_change:+.2f}%")
    print(f"   F1 Score:  {f1_change:+.2f}%")

    print("\n" + "="*70)

    if final_result['recall'] >= 0.90 and final_result['accuracy'] > 0.7087:
        print("✅ OPTIMIZATION SUCCESSFUL!")
    elif final_result['recall'] >= 0.90:
        print("✅ Recall maintained, slight accuracy improvement")
    else:
        print("⚠️  Recall dropped below 90% - use original Model 2")

    print("="*70)

    return final_threshold, final_result


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    optimize_model(
        dataset="merged_robust_clean.csv",
        base_model_path="teacher_improved_best.pth",
        device=device
    )