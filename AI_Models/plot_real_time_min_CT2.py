import argparse
import logging
import time
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from keras.models import load_model
from scipy.signal import butter, filtfilt, iirnotch
import joblib

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='Mindrove Plot',size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()


    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)

        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 4.0, 50.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 4.0, 60.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()

# def record_data(board_shim, num_points):
#     data = board_shim.get_current_board_data(num_points)
#     return data
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
        signalEnvelope[:,i] = np.convolve(signal[:,i], np.ones(window)/window, mode='same')
    return signalEnvelope

    
def main():

    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)


    params = MindRoveInputParams()
    board_shim = None
    try:
        board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        time.sleep(0.1)

        board_id = board_shim.get_board_id()
        exg_channels = BoardShim.get_exg_channels(board_shim.board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)
        update_speed_ms = 50
        scaler = joblib.load(r"C:\Personal_Programs\SURP\AI_Models\emg_scaler_Aiden.pkl")
        model = load_model(r"C:\Personal_Programs\SURP\AI_Models\cnn_flexion_Aiden.h5", compile=False)

        secs = input("Record Time(sec): ")
        window_size = int(secs)

        num_points = 0.2 * sampling_rate
        start_time = time.perf_counter()
        #data = np.zeros((0, 8))
        data = np.array([])
        i = 0

        while((time.perf_counter()-start_time)< window_size):
            while ((time.perf_counter()-start_time)<0.2):
                window_size = 4

            data = board_shim.get_board_data(100)
            print(data.shape)
            print(i)
            if data.shape[1] == 100:
                i = i + 1
                #data = np.append(data, temp)
                emg_channels = board_shim.get_emg_channels(board_id)
                acc_channels = board_shim.get_accel_channels(board_id)
                gyro_channels = board_shim.get_gyro_channels(board_id)
                for __, channel in enumerate(emg_channels):
                    DataFilter.remove_environmental_noise(data[channel], sampling_rate, NoiseTypes.SIXTY.value)


                data = np.transpose(data)
                inputData = np.reshape(data[0:101], shape=(-1, 8))
                #print(inputData)
                #print(inputData.shape)
                bandPassInputData = bandpass_filter(inputData)
                rectifiedInputData = rectified_data(bandPassInputData)
                finalInputData = envelope_data(rectifiedInputData)

                #scaler.fit(finalInputData)
                finalInputData = scaler.transform(np.reshape(finalInputData, shape=(-1,1)))

                usable_rows = finalInputData.shape[0] // 100 * 100
                finalInputData = finalInputData[:usable_rows]
                reshaped_input = finalInputData.reshape(-1, 100, 8)

                pred = model.predict(reshaped_input)

                #scaler.fit(pred)
                pred = scaler.inverse_transform(pred)

                print("Prediction: \n", pred)
                data = np.array([])
            else:
                exit()



            #print(data.shape)
            #print(emg_channels)
            #print(acc_channels)
            #print(gyro_channels)
        board_shim.release_session()
             
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim is not None and board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session() 


if __name__ == '__main__':
    main()
