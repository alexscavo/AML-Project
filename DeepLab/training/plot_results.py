import re
import matplotlib.pyplot as plt

# Path to your log file
LOG_FILE = "training/results/logs/training_DeepLab.log"

# Read the log file
with open(LOG_FILE, "r") as f:
    log_text = f.read()

# Regular expressions
train_pattern = r"Epoch \[(\d+)/\d+\] Training - Loss: ([\d.]+), mIoU: ([\d.]+), Accuracy: ([\d.]+)%"
val_pattern = r"Epoch \[(\d+)/\d+\] Validation - Loss: ([\d.]+), mIoU: ([\d.]+), Accuracy: ([\d.]+) %"

# Parse training data
train_data = re.findall(train_pattern, log_text)
epochs = [int(epoch) for epoch, *_ in train_data]
train_losses = [float(loss) for _, loss, _, _ in train_data]
train_mious = [float(miou) for _, _, miou, _ in train_data]
train_accuracies = [float(acc) for _, _, _, acc in train_data]

# Parse validation data
val_data = re.findall(val_pattern, log_text)
val_epoch_map = {
    int(epoch): (float(loss), float(miou), float(acc))
    for epoch, loss, miou, acc in val_data
}
val_losses = [val_epoch_map.get(epoch, (None, None, None))[0] for epoch in epochs]
val_mious = [val_epoch_map.get(epoch, (None, None, None))[1] for epoch in epochs]
val_accuracies = [val_epoch_map.get(epoch, (None, None, None))[2] for epoch in epochs]

# Plotting settings
plot_train_loss = False
plot_val_loss = False
plot_val_miou = False
plot_train_acc = True
plot_val_acc = True

# Plotting
plt.figure(figsize=(14, 8))

if plot_train_loss:
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", color="orange")

if plot_val_loss:
    plt.plot(epochs, val_losses, label="Val Loss", marker="s", color="red")

if plot_val_miou:
    plt.plot(epochs, val_mious, label="Val mIoU", marker="x", color="green")
    max_miou = max([v for v in val_mious if v is not None])
    plt.axhline(max_miou, linestyle='--', color='green', alpha=0.5, label=f"Max mIoU: {max_miou:.4f}")

if plot_train_acc:
    plt.plot(epochs, train_accuracies, label="Train Acc", marker="^", color="blue")

if plot_val_acc:
    plt.plot(epochs, val_accuracies, label="Val Acc", marker="v", color="purple")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("training_metrics_plot.png", dpi=300)
plt.show()
