from EMGData import MindroveBoard
import numpy as np
import sys
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
import time

class LiveEMGPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live EMG Plot (8 Channels)")

        # Set up the board
        self.board = MindroveBoard()
        self.board.connectBoard(DataType="EMG")
        self.board.board_shim.stop_stream()
        self.board.board_shim.start_stream()

        # Channel info
        self.channels = self.board.eeg_channels  # First 8 channels
        self.data_window = 500
        self.plot_lines = []

        # Colors for each channel
        self.colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w', 'orange']

        # Set up plot widget
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        for i in range(8):
            plot_line = self.plot_widget.plot(np.zeros(self.data_window), pen=pg.mkPen(self.colors[i], width=1))
            self.plot_lines.append(plot_line)

        self.plot_widget.addLegend()

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

    def update_plot(self):
        data = self.board.board_shim.get_current_board_data(self.data_window)
        if data is not None and data.shape[1] > 0:
            for i, ch in enumerate(self.channels):
                emg_channel_data = data[ch, -self.data_window:]
                self.plot_lines[i].setData(emg_channel_data)

    def closeEvent(self, event):
        print("Stopping board...")
        self.board.board_shim.stop_stream()
        self.board.board_shim.release_session()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LiveEMGPlot()
    win.show()
    sys.exit(app.exec())