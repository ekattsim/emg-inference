import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt

"""gyroY = pd.read_csv("C:\Personal_Programs\SURP\Test_Model_Data\GyroIt_1.csv", usecols= ["y-axis (deg/s)", "elapsed (s)"]).to_numpy()
accel = pd.read_csv("C:\Personal_Programs\SURP\Test_Model_Data\AccelIt_1.csv", usecols= ["x-axis (g)","y-axis (g)","z-axis (g)", "elapsed (s)"]).to_numpy()
#EMG = pd.read_csv("C:\Personal_Programs\SURP\Test_Model_Data\It_1.csv, ")

angle = 0
gyroAngle = 0
time = 0
angles = []
accelAngles = np.array([])
times = np.array([])
accelTimes = np.array([])
alpha = 0.98
pitch = 0
for i in range (accel.shape[0]):
    accel_pitch = math.atan2(-accel[i][0], math.sqrt(accel[i][1]**2 + accel[i][2]**2)) * (180 / math.pi)
    accelTimes = np.append(accelTimes, accel[i][0])
    accelAngles = np.append(accelAngles, accel_pitch)
for i in range(gyroY.shape[0]):
    if i == 0:
        time = 0
    else:
        time = gyroY[i][0]-gyroY[i-1][0]
    times = np.append(times, gyroY[i][0])
    gyroAngle += gyroY[i][1] * (time)
    allignedAccelAngles = np.interp(times, accelTimes, accelAngles)
    pitch = (alpha * (pitch + gyroY[i][1] * (time)) + (1 - alpha) * (allignedAccelAngles[i]))
    print(allignedAccelAngles[i])

    angles.append((pitch))

plt.plot(times, angles, label='Angle')
plt.xlabel('Time')
plt.ylabel('Angles')
plt.legend()
plt.show() """

class TestModelDataSorter():
    def __init__(self):
        self.angle = 0
        self.time = 0

        self.accelAngles = np.array([])
        self.accelTimes = np.array([])

        self.gyro = np.array([])
        self.accel = np.array([])
        self.emg = np.array([])
        self.emgFilter = np.array([])

    def ReadCSVFile(self, gyro_filepath: Path, accel_filepath: Path, emg_filepath: Path):
        fileFound = True
        if gyro_filepath.is_file():
            self.gyro = pd.read_csv(gyro_filepath, usecols= ["y-axis (deg/s)", "elapsed (s)"]).to_numpy()
        else:
            print("Gyro filepath not found")
            fileFound = False
        if accel_filepath.is_file():
            self.accel = pd.read_csv(accel_filepath, usecols= ["x-axis (g)","y-axis (g)","z-axis (g)", "elapsed (s)"]).to_numpy()
        else:
            print("Accel filepath not found")
            fileFound = False
        if emg_filepath.is_file():
            self.emg = pd.read_csv(emg_filepath, usecols= ["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6","Channel7","Channel8"]).to_numpy()
            self.emgFilter = pd.read_csv(emg_filepath, usecols= ["FilteredChannel1",	"FilteredChannel2",	"FilteredChannel3",	"FilteredChannel4",	"FilteredChannel5",	"FilteredChannel6",	"FilteredChannel7",	"FilteredChannel8",]).to_numpy()
        else:
            print("Emg filepath not found")
            fileFound = False
        if not fileFound:
            raise FileNotFoundError()
    def calcAccel(self) -> None:
        accel = self.accel
        for i in range (accel.shape[0]):
            accel_pitch = math.atan2(-accel[i][0], math.sqrt(accel[i][1]**2 + accel[i][2]**2)) * (180 / math.pi)

            self.accelTimes = np.append(self.accelTimes, accel[i][0])
            self.accelAngles = np.append(self.accelAngles, accel_pitch)
        
    def findAngles(self) -> tuple[np.array, np.array]:
        self.calcAccel()

        gyro = self.gyro
        alpha = 0.98
        pitch = 0

        angles = np.array([])
        times = np.array([])
        for i in range(self.gyro.shape[0]):
            if i == 0:
                time = 0
            else:
                time = gyro[i][0]-gyro[i-1][0]
            
            times = np.append(times, gyro[i][0])
            allignedAccelAngles = np.interp(times, self.accelTimes, self.accelAngles)
            pitch = (alpha * (pitch + gyro[i][1] * (time)) + (1 - alpha) * (allignedAccelAngles[i]))

            angles= np.append(angles, pitch)

        return angles, times
    
    def graphAngles(self):
        angles, times = self.findAngles()

        plt.plot(times, angles, label='Angle')
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.legend()
        plt.show()

Data = TestModelDataSorter()
gyroPath = Path("C:\Personal_Programs\SURP\Test_Model_Data\GyroIt_1.csv")
accelPath = Path("C:\Personal_Programs\SURP\Test_Model_Data\AccelIt_1.csv")
emgPath = Path("C:\Personal_Programs\SURP\Test_Model_Data\It_1.csv")

Data.ReadCSVFile(gyro_filepath=gyroPath, accel_filepath=accelPath, emg_filepath=emgPath)