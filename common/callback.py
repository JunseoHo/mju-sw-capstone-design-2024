import numpy as np
from keras.callbacks import Callback
from matplotlib import pyplot as plt


class TrainCallback(Callback):
    def __init__(self, validation_data, freq=1):
        super().__init__()
        self.validation_data = validation_data
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            x_val, y_val = next(iter(self.validation_data))
            predictions = self.model.predict(x_val)

            for prediction, ground_truth in zip(predictions, y_val):
                prediction_mask = np.zeros((256, 256))
                ground_truth_mask = np.zeros((256, 256))
                for i in range(256):
                    for j in range(256):
                        prediction_mask[i][j] = 1 if prediction[i][j] > 0.5 else 0
                        ground_truth_mask[i][j] = ground_truth[i][j]

                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(prediction_mask, cmap='binary')
                axes[0].set_title("Predicted Mask")
                axes[0].axis('off')
                axes[1].imshow(ground_truth_mask, cmap='binary')
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis('off')
                plt.show()
