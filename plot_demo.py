from train_val_loss_plot import TrainValLossPlot
from los_distribution_plot import LOSDistributionPlot
import numpy as np

# Dummy loss data
train_losses = [37, 24, 15, 9.6, 5.4, 3.5, 2.8, 2.6, 2.1, 2.0, 1.9, 1.9, 1.8, 1.7, 1.7, 1.6, 1.6, 1.5, 1.6, 1.4, 1.6, 1.4, 1.5]
val_losses = [50.8, 38.8, 29.9, 20.1, 10.9, 8.9, 5.1, 2.5, 2.2, 1.0, 1.4, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 0.9, 1.1, 1.0, 1.2, 1.0, 1.1]

# Dummy LOS distribution
los_labels = np.random.normal(loc=5, scale=2, size=1000).clip(0, 15)

# Generate plots
TrainValLossPlot(train_losses, val_losses).plot()
LOSDistributionPlot(los_labels).plot()
