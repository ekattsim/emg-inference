import time
import mindrove
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRoveError
import numpy as np
import signal
import argparse

class MindroveBoard():
    def __init__(self):
        #BoardShim.enable_dev_board_logger()
        self.params = MindRoveInputParams()
        self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        self.board_shim = BoardShim(self.board_id, self.params)
        self.is_eeg = False
        self.is_accel = False
        self.accel_channels = None
        self.eeg_channels = None
        self.gyro_channels = None

        signal.signal(signal.SIGINT, self.handle_exit)
        self.running = True

    def handle_exit(self, signum, frame):
        print("\nReceived exit signal. Stopping...")
        self.running = False

    def connectBoard(self, DataType: str):
        """
        Connectes to the board based on MAC Address. Also configures the board based on desired Datatype
        :param DataType: The desired data type to configure the board for
        """

        try:
            board_id = self.board_id
            params = self.params
            board_shim = self.board_shim

            params.mac_address = "eb1580"
            board_shim.prepare_session()
            board_shim.start_stream(450000)
            print("Board Connected")

        except MindRoveError:
            print(MindRoveError)
            exit()

        if DataType.lower() == "emg" or DataType.lower() == "eeg":
            self.eeg_channels = board_shim.get_eeg_channels(board_id)
            self.accel_channels = board_shim.get_accel_channels(board_id)
            self.gyro_channels = board_shim.get_gyro_channels(board_id)
            self.is_eeg = True
        elif DataType.lower() == "accel":
            self.eeg_channels = board_shim.get_eeg_channels(board_id)
            self.accel_channels = board_shim.get_accel_channels(board_id)
            self.gyro_channels = board_shim.get_gyro_channels(board_id)
            self.is_accel = True

        self.configureBoard()

    def configureBoard(self):
        """
        Starts board configuration
        """
        is_eeg = self.is_eeg
        is_accel = self.is_accel
        if is_eeg:
            try:
                self.board_shim.config_board(mindrove.MindroveConfigMode.EEG_MODE)
                print("Device configured in EEG mode")
            except:
                print("Could not configure")
                exit()
        elif is_accel:
            try:
                self.board_shim.config_board(mindrove.MindroveConfigMode.EEG_MODE)
                print("Device configured in ACCEL mode")
            except:
                print("Could not configure")
                exit()
        else:
            print("Invalid data type")

    def getDataInstance(self, numPoints):
        """
        gets an instance of data from the MindroveBoard

        :param numPoints: Numbers of data points eeg_data list should have based on sampling rate
        """
        board_shim = self.board_shim
        is_eeg = self.is_eeg
        is_accel = self.is_accel

        eeg_data = np.array([])
        accel_data = np.array([])
        gyro_data = np.array([])
        data = board_shim.get_current_board_data(numPoints)
        if data is not None:
            if is_eeg:
                eeg_data = data[self.eeg_channels]
                accel_data = data[self.accel_channels]
                gyro_data = data[self.gyro_channels]
            elif is_accel:
                eeg_data = data[self.eeg_channels]
                accel_data = data[self.accel_channels]
                gyro_data = data[self.gyro_channels]
        if (eeg_data.size and accel_data.size and gyro_data.size) is not 0:
            return eeg_data[:,0], accel_data[:,0], gyro_data[:,0]
        else:
            print("Error: No data received, check network connection")
            return 0, 0, 0

    def sendData(self, ReceiveTime, Print: bool):
        """
        Sends data received from the MindroveBoard

        :param ReceiveTime: The amount of time data is to be Received and sent (seconds).
        :param Print: Print data to terminal.
        """
        board_shim = self.board_shim

        eeg_data = np.array([])
        accel_data = np.array([])
        gyro_data = np.array([])

        sampling_rate = board_shim.get_sampling_rate(self.board_id)
        startTime = time.time()
        try:
            ReceiveTime = int(ReceiveTime)
            num_points = ReceiveTime * sampling_rate
        except:
            print("Error: Please enter a number")
            exit()
        
        print("Sending recived data. Press ctrl+C to stop early")
        while time.time() - startTime < ReceiveTime and self.running:
            eeg_data, accel_data, gyro_data = self.getDataInstance(num_points)
        
        if Print:
            print("EMG data \n", eeg_data)
            print("Accel data \n", accel_data)
            print("Gyro data \n", gyro_data)
        return eeg_data, accel_data, gyro_data

    def sendContinuousData(self, Print: bool):
        """
        Sends data received from the MindroveBoard until terminated

        :param Print: Print data to terminal.
        """

        eeg_data = np.array([])
        accel_data = np.array([])
        gyro_data = np.array([])

        print("Sending recived data. Press ctrl+C to stop early")
        while self.running:
            eeg_data, accel_data, gyro_data = self.getDataInstance(numPoints=1)
            if Print:
                print("EMG data \n", eeg_data)
                print("Accel data \n", accel_data)
                print("Gyro data \n", gyro_data)
        return eeg_data, accel_data, gyro_data
    

def main():
    parser = argparse.ArgumentParser(description="Mindrove board interface")
    parser.add_argument("mode", choices=["timed", "continuous"], help="Choose the data reception mode")
    parser.add_argument("--datatype", type=str, default="eeg", help="Data type: eeg, emg, or test")
    parser.add_argument("--time", type=int, default=10, help="Time (in seconds) to receive data (only for 'receive' mode)")
    parser.add_argument("--print",type=bool, default=True, help="Should received data be printed to terminal")
    args = parser.parse_args()

    minddrove_board = MindroveBoard()
    minddrove_board.connectBoard(DataType=args.datatype)
    time.sleep(0.1) #Adding a delay allows the EMG sensor time to fully connect and stops the first responses from being errors

    if args.mode == "timed":
        minddrove_board.sendData(ReceiveTime=str(args.time), Print=args.print)
    elif args.mode == "continuous":
        minddrove_board.sendContinuousData(Print=args.print)
        

if __name__ == "__main__":
    main()

#python3.9 EMGData.py timed/continuous --datatype=EMG/test --time=10 --print=True

