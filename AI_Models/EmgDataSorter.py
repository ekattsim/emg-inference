from EMG_Data import MindroveBoard
import signal
import numpy as np
import tty
import termios
import sys
import select
import pandas as pd


class DataSorter():
    """
    Sort EMG data into different docuemnts based on keys pressed
    """
    def __init__(self) -> None:
        self.running = True
        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, signum, frame):
        print("\nReceived exit signal. Stopping...") 
        self.running = False

    def trackInput(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                return sys.stdin.read(1)
            else:
                return []
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None

    def recordData(self):
        """
        Records the data sent by the EMG sensors. Data is seperated into different lists based on key press. If no key is being pressed, data is not saved
        """

        board = MindroveBoard()
        board.connectBoard("EMG")

        lst1 = np.empty((0,8))
        lst2 = np.empty((0,8))
        while True:
            key = self.trackInput()
            emgData, accelData, gyroData = board.getDataInstance(1)
            emgData = np.array([[float(x) for x in emgData]])
            if key == 'w':
                lst1 = np.append(lst1, emgData, axis=0)
            if key == 'e':
                lst2 = np.append(lst2, emgData, axis=0)
            print(f"pressed {key}")
            if key == 'q':
                print("Ending Recording")
                break
        return lst1, lst2

    def outputData(self, Data, Name: str):
        """
        Outputs data to csv document.

        :param Data: the data to be output
        :param Name: the file name
        """
        df = pd.DataFrame(Data)
        df.to_csv(Name, index=False, header=False)




data = DataSorter()
lst1, lst2 = data.recordData()
data.outputData(lst1, "Raised")
data.outputData(lst2, "Lowered")
