import asyncio
import logging
import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow.keras.models import load_model

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# Import your RoboticGloveController. Make sure the path is correct.
# Assuming 'Bluetooth_Communication' is in the parent directory.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Bluetooth_Communication.glove_controller import RoboticGloveController

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


class RealTimeGloveControl:
    """
    A class to handle the real-time EMG data collection, processing,
    and robotic glove control pipeline.
    """

    def __init__(self):
        # --- Configuration ---
        self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.emg_channels = BoardShim.get_emg_channels(self.board_id)

        # The number of data points required for one prediction by the model
        self.window_samples = 100

        self.data_buffer = np.array([[] for _ in self.emg_channels]).T
        self.counter = 0

        # --- Device and Model Initialization ---
        self.board_shim = None
        self.glove = RoboticGloveController()
        self.model = None
        self.infer = None
        self.scaler = None

    def _load_models_and_scaler(self):
        """Loads the Keras model and the scaler."""
        try:
            # stop tf from pre-allocating like 90% memory
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True) # we run into OOM errors without this
                logging.info("Enabled GPU memory growth")

            logging.info("Loading TF-TRT optimized model and scaler...")
            # Load the TF-TRT SavedModel
            trt_model_dir = "./models/tftrt_saved_model"
            self.model = tf.saved_model.load(trt_model_dir)
            self.infer = self.model.signatures["serving_default"]
            logging.info("TF-TRT model loaded.")

            # Load scaler
            scaler_path = "../AI_Models/emg_scaler_Aiden.pkl"
            self.scaler = joblib.load(scaler_path)
            logging.info("Scaler loaded.")
        except Exception as e:
            logging.error(f"Failed to load model or scaler: {e}")
            raise

    def _initialize_mindrove(self):
        """Initializes and prepares the MindRove board session."""
        logging.info("Initializing MindRove board...")
        params = MindRoveInputParams()
        # You can add specific params here if needed, e.g., params.serial_port
        self.board_shim = BoardShim(self.board_id, params)
        self.board_shim.prepare_session()
        logging.info("MindRove board prepared.")

    async def _connect_devices(self):
        """Connects to both MindRove and the robotic glove."""
        logging.info("Connecting to the robotic glove...")
        await self.glove.connect()
        if self.glove.is_connected:
            logging.info("Robotic glove connected successfully.")
        else:
            raise ConnectionError("Failed to connect to the robotic glove.")
        self._initialize_mindrove()

    def _process_window(self, data_window: np.array):
        """
        Takes a window of EMG data and runs the full inference pipeline.
        """
        # 1. Pre-process the data
        bandpassed = bandpass_filter(data_window, fs=self.sampling_rate)
        rectified = rectified_data(bandpassed)
        enveloped: np.ndarray = envelope_data(rectified, fs=self.sampling_rate)

        # 2. Scale the data
        # Reshape for scaler which expects (n_samples, 1)
        scaled_data = self.scaler.transform(enveloped.flatten().reshape(-1, 1))

        # 3. Reshape for the CNN model: (batch_size, window_size, channels)
        reshaped_input = scaled_data.reshape(
            1, self.window_samples, len(self.emg_channels))

        # 4. Predict
        input_tensor = tf.constant(reshaped, dtype=tf.float32)
        prediction = self.infer(input_tensor)
        pred_value = list(prediction.values())[0].numpy()

        # 5. Inverse transform the prediction to get the original scale
        inversed_prediction = self.scaler.inverse_transform(pred_value)

        # 6. Convert prediction to a servo angle that's between 0-180
        angle = round(int(inversed_prediction[0][0]) + 45)

        return angle

    async def run(self):
        """
        The main real-time loop for data collection, prediction, and control.
        """
        self._load_models_and_scaler()
        await self._connect_devices()

        # WARM-UP STEP ---
        logging.info("Warming up the model...")
        # Create a dummy input tensor with the same shape as your real data.
        # Shape: (batch_size, window_samples, num_channels)
        dummy_input = np.zeros(
            (self.window_samples, len(self.emg_channels)), dtype=np.float32)
        # The first prediction will be slow, but it happens here, before the loop.
        _ = self._process_window(dummy_input)
        logging.info(
            "Model is warmed up. Starting real-time inference loop...")
        # --- END WARM-UP ---

        logging.info("Starting real-time inference loop...")

        self.board_shim.start_stream(450000)
        start_time = time.time()

        try:
            while True:
                # Get new data from the board
                elapsed = time.time() - start_time
                logging.info(f"""[{elapsed:.2f}s] Ringbuffer: {
                             self.board_shim.get_board_data_count()}""")

                # accumulate exactly enough samples for the first prediction
                if len(self.data_buffer) < self.window_samples:
                    new_data = self.board_shim.get_board_data(
                        self.window_samples - len(self.data_buffer))
                else:
                    new_data = self.board_shim.get_board_data()

                # if we have new data, append, slide, and process
                if new_data.shape[1] > 0:
                    emg_data = new_data[self.emg_channels, :].T
                    logging.info(f"Fetch size: {emg_data.shape[0]}")

                    # Append new data to our buffer
                    self.data_buffer = np.vstack([self.data_buffer, emg_data])

                    # If we have enough data for a full window, process it
                    if len(self.data_buffer) >= self.window_samples:
                        # Get the most recent window for processing
                        recent_window_index = len(self.data_buffer) - self.window_samples
                        processing_window = self.data_buffer[recent_window_index:, :]

                        # slide the window
                        self.data_buffer = processing_window

                        # Run the prediction pipeline
                        angle = self._process_window(processing_window)
                        self.counter += 1
                        elapsed = time.time() - start_time
                        logging.info(f"""[{elapsed:.2f}s] Prediction {
                                    self.counter}: {angle} degrees""")

                        # Send command to the glove
                        if self.glove.is_connected:
                            await self.glove.set_servo_angle(1, angle)

                # Short pause to prevent hogging CPU and allow other async tasks
                await asyncio.sleep(0.02)

        except KeyboardInterrupt:
            logging.info("Caught KeyboardInterrupt. Shutting down.")
        except Exception as e:
            logging.error(f"An error occurred during the main loop: {e}")
        finally:
            logging.info("Cleaning up and disconnecting devices...")
            if self.board_shim and self.board_shim.is_prepared():
                self.board_shim.stop_stream()
                self.board_shim.release_session()
                logging.info("MindRove session released.")
            if self.glove.is_connected:
                await self.glove.disconnect()
                logging.info("Glove disconnected.")


async def main():
    """Main function to run the controller."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    controller = RealTimeGloveControl()
    await controller.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"Program failed to run: {e}")
