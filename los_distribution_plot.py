import matplotlib.pyplot as plt

class LOSDistributionPlot:
    def __init__(self, los_labels):
        self.los_labels = los_labels

    def plot(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.los_labels, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Length of Stay (days)")
        plt.ylabel("Number of Samples")
        plt.title("Distribution of LOS Labels")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
