import tf2onnx
import tensorflow as tf
import onnx
import joblib
import time
from tensorflow import keras
from keras.models import load_model
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import pandas as pd
import asyncio

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Bluetooth_Communication.glove_controller import RoboticGloveController


def bandpass_filter(signal, fs=500, low=10, high=180, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


def rectified_data(signal):
    return np.abs(signal)


def envelope_data(signal):
    window = int(0.05 * 500)  # 100 ms window
    signalEnvelope = np.zeros(signal.shape)
    for i in range(8):
        signalEnvelope[:, i] = np.convolve(
            signal[:, i], np.ones(window)/window, mode='same')
    return signalEnvelope


async def main():
    model = load_model(
        os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "cnn_flexion_Aiden.h5")),
        compile=False)
    scaler = joblib.load(
        os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "emg_scaler_Aiden.pkl")))
    model.summary()

    data = pd.read_csv(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                     "test_data_Notch.csv")), header=None)

    EMGdata = data.to_numpy()

    glove = RoboticGloveController()
    try:
        await glove.connect()
    except:
        print("Connection failed")
        exit()

    print("Robot hand connection succesful!!")

    user_input = input("Save?(Y/N)")
    if user_input == "Y":
        input_signature = [
            tf.TensorSpec(input_tensor.shape, input_tensor.dtype,
                          name=input_tensor.name)
            for input_tensor in model.inputs
        ]

        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13
        )

        onnx.save(onnx_model, "New_Onnx_Model_Aiden.onnx")

    elif user_input == "N":
        inputData = np.array([])
        for data in EMGdata[1:5001]:
            number = [float(value) for value in data]
            numpyNumber = np.array(number)
            inputData = np.append(inputData, [numpyNumber[0:8]]).flatten()
            inputData = np.reshape(inputData, shape=(-1, 8))
            # print(inputData)
            # print(inputData.shape)
        bandPassInputData = bandpass_filter(inputData)
        rectifiedInputData = rectified_data(bandPassInputData)
        finalInputData = envelope_data(rectifiedInputData)

        # scaler.fit(finalInputData)
        finalInputData = scaler.transform(
            np.reshape(finalInputData, shape=(-1, 1)))

        usable_rows = finalInputData.shape[0] // 100 * 100
        finalInputData = finalInputData[:usable_rows]
        reshaped_input = finalInputData.reshape(-1, 100, 8)

        pred = model.predict(reshaped_input)

        # scaler.fit(pred)

        pred = np.array(pred)
        pred = scaler.inverse_transform(pred)
        pred = pred + 90

        for i in pred:
            await glove.set_servo_angle(1, i)
            time.sleep(0.05)

        await glove.disconnect()
        print("glove disconnected")

        # print("Prediction: \n", pred)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nProgram failed: {e}")
