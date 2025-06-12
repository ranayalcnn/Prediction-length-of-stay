import matplotlib.pyplot as plt

class TrainValLossPlot:
    def __init__(self, train_losses, val_losses):
        self.train_losses = train_losses
        self.val_losses = val_losses

    def plot(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
