import asyncio
import logging
import time
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt
import numpy as np
import tensorflow as tf

# --- Pre-processing Functions ---

def bandpass_filter(signal, fs=500, low=10, high=180, order=4):
    """Applies a bandpass filter to the signal."""
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


def rectified_data(signal):
    """Rectifies the signal."""
    return np.abs(signal)


def envelope_data(signal, fs=500):
    """Creates an envelope of the signal."""
    window_size = int(0.05 * fs)  # 50 ms window
    envelope = np.zeros(signal.shape)
    for i in range(signal.shape[1]):
        envelope[:, i] = np.convolve(
            signal[:, i], np.ones(window_size) / window_size, mode='same'
        )
    return envelope

class InferencePipelineTester:
    def __init__(self):
        self.window_samples = 100
        self.num_channels = 8
        self.sampling_rate = 500
        self.model = None
        self.scaler = None
        self.trt_infer = None

    def _load_model_and_scaler(self):
        # stop tf from pre-allocating like 90% memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True) # we run into OOM errors without this
            logging.info("Enabled GPU memory growth")

        logging.info("Loading TF-TRT optimized model and scaler...")
        # Load the TF-TRT SavedModel
        trt_model_dir = "./models/tftrt_saved_model"
        self.trt_model = tf.saved_model.load(trt_model_dir)
        self.trt_infer = self.trt_model.signatures["serving_default"]
        logging.info("TF-TRT model loaded.")

        # Load scaler
        scaler_path = "../AI_Models/emg_scaler_Aiden.pkl"
        self.scaler = joblib.load(scaler_path)
        logging.info("Scaler loaded.")

    def _generate_fake_emg_data(self, duration_sec=10):
        """Simulate EMG data for a given duration."""
        total_samples = int(duration_sec * self.sampling_rate)
        return np.random.randn(total_samples, self.num_channels).astype(np.float32)

    def _process_window(self, data_window):
        start = time.time()
        bandpassed = bandpass_filter(data_window, fs=self.sampling_rate)
        rectified = rectified_data(bandpassed)
        enveloped = envelope_data(rectified, fs=self.sampling_rate)
        scaled = self.scaler.transform(enveloped.flatten().reshape(-1, 1))
        reshaped = scaled.reshape(1, self.window_samples, self.num_channels)

        tf_input = tf.constant(reshaped, dtype=tf.float32)
        prediction = self.trt_infer(tf_input)
        # TF-TRT returns a dict of outputs; grab first key
        pred_value = list(prediction.values())[0].numpy()
        latency = (time.time() - start) * 1000  # ms

        return pred_value, latency

    async def run_test(self, duration_sec=10):
        logging.info(f"Starting inference benchmark for {duration_sec}s...")
        self._load_model_and_scaler()
        fake_data = self._generate_fake_emg_data(duration_sec)
        loop_start = time.time()

        iteration = 0
        while time.time() - loop_start < duration_sec:
            # Simulate streaming
            chunk = fake_data[iteration*self.window_samples:(iteration+1)*self.window_samples]
            if len(chunk) < self.window_samples:
                break

            prediction, latency = self._process_window(chunk)

            logging.info(
                f"[{iteration}] Inference latency: {latency:.2f} ms | "
                f"Pred: {prediction[0][0]:.4f}"
            )

            iteration += 1
            await asyncio.sleep(0.02)

        logging.info("Test complete.")

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    tester = InferencePipelineTester()
    await tester.run_test(duration_sec=30)

if __name__ == "__main__":
    asyncio.run(main())
