# %%
from preprocessing_pipeline import load_data
from models import MLBMatchupPredictor, MLBMatchupPredictorTemporal

import torch
import torch.nn.functional as F

from loss import GraphCalibrationLoss, compute_class_weights

from sklearn.metrics import classification_report, confusion_matrix
from evaluation_metrics import (
    plot_confusion_matrix,
    reliability_diagram,
    one_v_rest_roc_auc,
    compute_ece,
    brier_score,
    plot_loss_curve,
    plot_prob_distribution,
    plot_one_vs_rest_reliability_diagrams,
)

import pandas as pd
import numpy as np

# %%

train_data, val_data, test_data, label_encoder, test_label_encoder = load_data(
    "simple_outcome"
)



# %%
model_name = "gat_cross_entropy_loss_4_outcomes"
# model_name = "gat_cross_entropy_loss_8_outcomes"
# model_name = "gat_gcl_4_outcomes"
# model_name = "gat_gcl_8_outcomes"
# model_name = "temporal_gat_cross_entropy_loss_4_outcomes"
# model_name = "temporal_gat_cross_entropy_loss_8_outcomes"
# model_name = "temporal_gat_gcl_4_outcomes"
# model_name = "temporal_gat_gcl_8_outcomes"

# Update model and training loop
metadata = {
    "pitcher": train_data["pitcher"].x.shape[1],
    "batter": train_data["batter"].x.shape[1],
}

num_classes = len(label_encoder.classes_)

model = MLBMatchupPredictor(
    metadata=metadata,
    hidden_channels=128,
    edge_dim=train_data["pitcher", "faces", "batter"].edge_attr.shape[1],
    num_classes=num_classes,
).to("cpu")

loss_fn = torch.nn.CrossEntropyLoss(
    weight=compute_class_weights(train_data["pitcher", "faces", "batter"].y)
)

# loss_fn = GraphCalibrationLoss(gamma=0.01)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Track best validation performance
best_val_brier_score = 1
best_epoch = 0
patience = 20  # Early stopping patience
epochs_no_improve = 0

# Create lists to store metrics for later plotting
train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_brier_scores = []
val_brier_scores = []
epochs_list = []

for epoch in range(200):
    # Training
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    y = train_data["pitcher", "faces", "batter"].y
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(val_data)
        val_y = val_data["pitcher", "faces", "batter"].y
        val_loss = loss_fn(val_out, val_y)
        val_pred = val_out.argmax(dim=1)
        val_acc = (val_pred == val_y).float().mean()
        val_brier_score = brier_score(val_out, val_y, num_classes)

        # Training metrics
        train_pred = out.argmax(dim=1)
        train_acc = (train_pred == y).float().mean()
        train_brier_score = brier_score(out, y, num_classes)

    # Store metrics for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc.item())
    val_accs.append(val_acc.item())
    train_brier_scores.append(train_brier_score.item())
    val_brier_scores.append(val_brier_score.item())

    epochs_list.append(epoch)

    # Print metrics
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Train Brier Score: {train_brier_score:.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f} | Val Brier Score: {val_brier_score:.4f}"
        )

    # Check model improvement 
    if val_brier_score < best_val_brier_score:
        best_val_brier_score = val_brier_score
        best_epoch = epoch
        epochs_no_improve = 0
        # Save best model
        torch.save(model.state_dict(), f"saved_models/{model_name}/best_model.pt")

metrics_df = pd.DataFrame(
    {
        "epoch": epochs_list,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_brier_score": train_brier_scores, 
        "val_brier_score": val_brier_scores, 
    }
)

# Save to CSV
metrics_df.to_csv(f"saved_models/{model_name}/training_metrics.csv", index=False)

 # %%
# Load best model for testing
model.load_state_dict(torch.load(f"saved_models/{model_name}/best_model.pt"))
model.eval()

# Evaluate on test set
with torch.no_grad():
    test_out = model(test_data)
    test_y = test_data["pitcher", "faces", "batter"].y
    test_pred = test_out.argmax(dim=1)
    test_acc = (test_pred == test_y).float().mean()
    test_loss = loss_fn(test_out, test_y)

    # Calculate additional metrics
    test_pred_np = test_pred.cpu().numpy()
    test_y_np = test_y.cpu().numpy()

    # Classification report
    print("\nTest Performance:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(
        classification_report(
            test_y_np, test_pred_np, target_names=label_encoder.classes_
        )
    )

    # Save classification report as latex table
    report_dict = classification_report(
        test_y_np, test_pred_np, target_names=label_encoder.classes_, output_dict=True
    )

    # Convert to DataFrame
    df_report = pd.DataFrame(report_dict).transpose()

    # Optional: round for cleaner LaTeX
    df_report = df_report.round(2)

    # Save to LaTeX
    df_report.to_latex(f"figures/{model_name}/classification_report.tex")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(test_y_np, test_pred_np)
    print(conf_matrix)
    plot_confusion_matrix(test_y_np, test_pred_np, test_label_encoder.classes_, model_name)

    # Reliability diagram
    print("\nReliability Diagram")
    test_probs = torch.softmax(test_out, dim=1)
    test_conf, test_pred = torch.max(test_probs, dim=1)
    test_acc = (test_pred == test_y).float()

    reliability_diagram(test_conf, test_acc, model_name)

    # One vs Rest Reliability diagram 
    plot_one_vs_rest_reliability_diagrams(test_y, test_probs, test_label_encoder.classes_, model_name)

    # One vs Rest ROC Curve
    print("\nROC")
    one_v_rest_roc_auc(test_out, test_y, test_label_encoder, model_name)

    # Brier Score and ECE
    cal_metrics_df = pd.DataFrame(
        {
            "Metric": ["Brier Score", "ECE"],
            "Value": [
                brier_score(test_out, test_y, num_classes=num_classes).item(),
                compute_ece(test_probs, test_y).item(),
            ],
        }
    )

    cal_metrics_df.to_csv(f"figures/{model_name}/calibration_metrics.csv", index=False)

    # Probability distribution 
    plot_prob_distribution(test_probs, test_y, test_label_encoder, model_name)

plot_loss_curve(epochs_list, train_losses, val_losses, model_name)


# %%
