import time
import mindrove
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRoveError
import numpy as np
from matplotlib import pyplot as plt

try:
    BoardShim.enable_dev_board_logger()

    params = MindRoveInputParams()
    params.mac_address = "eb1580"
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    board_shim = BoardShim(board_id, params)
    board_shim.prepare_session()
    board_shim.start_stream(450000)

    eeg_channels = board_shim.get_eeg_channels(board_id)
    accel_channels = board_shim.get_accel_channels(board_id)
    sampling_rate = board_shim.get_sampling_rate(board_id)
    n_package = 0
    print("Device ready")
except:
    print("No Board Detected")
    exit()

if board_shim is not None:
    # Enable EEG mode and read 5 seconds of EEG data
    #"""
    board_shim.config_board(mindrove.MindroveConfigMode.EEG_MODE) # EEG_MODE 
    print("Device configured in EEG mode")
    print("Read 5 seconds of EEG data")
    
    time.sleep(2)
    #board_shim.release_session()

    window_size = 5 # seconds
    num_points = window_size * sampling_rate
    currentTime = time.time()
    eeg_data = np.array([])
    accel_data = np.array([])

    data = board_shim.get_current_board_data(num_points)
    while time.time() - currentTime < 5:
        if (board_shim.get_board_data_count() >= num_points) and (data is not None):
            eeg_data = data[eeg_channels] # output of shape (8, num_of_samples) ## Beware that depending on the electrode configuration, some channels can be *inactive*, resulting in all-zero data for that particular channel
            accel_data = data[accel_channels] # output of shape (3, num_of_samples)
            # process data, or print it out
    eeg_data = np.array(eeg_data)
    accel_data = np.array(accel_data)

    X = np.arange(len(eeg_data[0]))
    for i in range(eeg_data[:, 0].size):
        plt.plot(X, eeg_data[i])
        label = f"Figure, {i+1}"
        plt.title(label)
        plt.show()