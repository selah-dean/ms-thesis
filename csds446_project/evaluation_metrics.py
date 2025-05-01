import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Plots a confusion matrix as a heatmap with percentage values, where the values represent the percentage of correct predictions for each class.

    Args:
        y_true: True labels of the samples.
        y_pred: Predicted labels from the model.
        class_names: List of class labels for the axes.
        model_name: Name of the model to create a folder for saving the plot.

    Saves the confusion matrix plot as a PNG image and displays it.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentages (normalize by row)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(6, 6))

    # Create a heatmap with percentages only
    ax = sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,  # ✅ Removes the legend/color bar
    )

    # Add % symbol to annotations
    for text in ax.texts:
        text.set_text(text.get_text() + "%")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/confusion_matrix.png")
    plt.show()


def reliability_diagram(confidences, accuracies, model_name, n_bins=10):
    """
    Plots a reliability diagram to evaluate the calibration of predicted probabilities.

    Parameters:
        confidences: Predicted confidence scores (probabilities).
        accuracies: True/false values indicating correct/incorrect predictions.
        model_name: String, name of the model to create a folder for saving the plot.
        n_bins: number of bins to divide the confidence values for plotting.

    Saves the reliability diagram plot as a PNG image and displays it.
    """
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_bounds[:-1]
    bin_uppers = bin_bounds[1:]

    bin_confidences = []
    bin_accuracies = []

    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()

    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > lower) & (confidences <= upper)
        if np.any(mask):
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(bin_confidences, bin_accuracies, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/reliability_diagram.png")
    plt.show()


def one_v_rest_roc_auc(test_out, test_y, label_encoder, model_name):
    """
    Plots the ROC curve for a one-vs-rest classification task and calculates the AUC score for each class.

    Args:
        test_out: Output logits from the model.
        test_y: True labels for the test set.
        label_encoder: Encoder used to encode class labels.
        model_name: Name of the model to create a folder for saving the plot.

    Saves the ROC curve plot as a PNG image and displays it.
    """

    y_score = torch.softmax(test_out, dim=1).detach().cpu().numpy()

    # Binarize the labels for OvR
    y_true_bin = label_binarize(
        test_y, classes=list(range(len(label_encoder.classes_)))
    )

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(6, 6))

    for i, name in enumerate(label_encoder.classes_):
        RocCurveDisplay(
            fpr=fpr[i], tpr=tpr[i], roc_auc=roc_auc[i], estimator_name=name
        ).plot(ax=plt.gca())

    plt.title("One-vs-Rest ROC Curves")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/roc_curves.png")
    plt.show()


def brier_score(y_pred, y_true, num_classes):
    """
    Calculates the Brier score for multi-class classification.

    Args:
        y_pred: Predicted logits from the model.
        y_true: True class labels.
        num_classes (int): Number of classes for one-hot encoding the labels.

    Returns:
        The Brier score for the predicted probabilities.
    """
    probs = torch.softmax(y_pred, dim=1)
    # One-hot encode the targets
    y_onehot = F.one_hot(y_true, num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - y_onehot) ** 2, dim=1))


def compute_ece(y_pred, y_true, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) for the model's predicted probabilities.

    Parameters:
        y_pred: Predicted logits from the model.
        y_true: True class labels.
        n_bins: Number of bins to divide the confidence values for calculating ECE.

    Returns:
        The calculated ECE score.
    """
    probs = torch.softmax(y_pred, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == y_true).float()

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            bin_weight = mask.float().mean()
            ece += torch.abs(avg_conf - avg_acc) * bin_weight

    return ece


def plot_loss_curve(epochs_list, train_losses, val_losses, model_name):
    """
    Plots the training and validation loss curves over epochs.

    Args:
    - epochs_list: The epoch numbers.
    - train_losses: The training loss values for each epoch.
    - val_losses: The validation loss values for each epoch.
    - model_name: Name of the model to create a folder for saving the plot.

    Saves the loss curve plot as a PNG image and displays it.
    """
    # Plot losses
    plt.figure(figsize=(6, 6))
    plt.plot(epochs_list, train_losses, label="Train Loss")
    plt.plot(epochs_list, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/training_curves.png")
    plt.show()


def plot_prob_distribution(test_probs, test_y, label_encoder, model_name):
    """
    Plots the predicted probability distribution for each class, comparing it to the true class proportions.

    Args:
        test_probs: Predicted probabilities for each class.
        test_y: True labels for the test set.
        label_encoder: Encoder used to encode class labels.
        model_name: Name of the model to create a folder for saving the plot.

    Saves the probability distribution plot as a PNG image and displays it.
    """
    keys, counts = np.unique(test_y, return_counts=True)
    total = counts.sum()
    true_percentages = counts / total

    # test_probs = torch.softmax(test_out, dim=1)
    colors = ["blue", "green", "orange", "red"]
    for i in range(4):
        sns.kdeplot(
            test_probs[:, i],
            label=label_encoder.classes_[i],
            color=colors[i],
            fill=True,
            alpha=0.3,
        )
        plt.axvline(true_percentages[i], color=colors[i], linestyle="--", linewidth=2)

    plt.title("Predicted Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/{model_name}/probability_distribution.png")
    plt.show()


def plot_one_vs_rest_reliability_diagrams(
    y_true, y_pred_proba, class_names, model_name, n_bins=10
):
    """
    Plots one-vs-rest reliability diagrams for each class, comparing predicted confidence to observed accuracy.

    Args:
        y_true: True labels for the test set.
        y_pred_proba: Predicted probabilities for each class.
        class_names: List of class names.
        model_name: Name of the model to create a folder for saving the plot.
        n_bins: Number of bins for the reliability curve.

    Saves the reliability diagram plot as a PNG image and displays it.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(16, 4))

    y_true_one_hot = F.one_hot(y_true, num_classes=n_classes).float()

    for i, class_name in enumerate(class_names):
        ax = axes[i]

        # Convert to one-vs-rest
        y_true_binary = y_true_one_hot[:, i]
        y_pred_binary = y_pred_proba[:, i]

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true_binary, y_pred_binary, n_bins=n_bins, strategy="uniform"
        )

        # Plot
        ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "s-", label="Model")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{class_name}")
        ax.legend(loc="lower right")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/one_vs_rest_reliability_diagram.png")
    plt.show()
