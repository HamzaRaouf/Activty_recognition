import numpy as np 
ts_msAcc = np.load("testMSAccelerometer.npy")
ts_msGyr = np.load("testMSGyroscope.npy")
ts_labels = np.load("testLabels.npy")

OPEN_DOOR = 20
RUB_HANDS = 36

ts_labels_OPEN_DOOR_idx = ts_labels == OPEN_DOOR
ts_labels_RUB_HANDS_idx = ts_labels == RUB_HANDS

ts_msAcc_OPEN_DOOR = ts_msAcc[ts_labels_OPEN_DOOR_idx]
ts_msGyr_OPEN_DOOR = ts_msGyr[ts_labels_OPEN_DOOR_idx]

ts_msAcc_RUB_HANDS = ts_msAcc[ts_labels_RUB_HANDS_idx]
ts_msGyr_RUB_HANDS = ts_msGyr[ts_labels_RUB_HANDS_idx]

ts_labels_OPEN_DOOR = ts_labels[ts_labels_OPEN_DOOR_idx]
ts_labels_RUB_HANDS = ts_labels[ts_labels_RUB_HANDS_idx]

ts_msAcc_Two_Activities = np.concatenate((ts_msAcc_OPEN_DOOR, ts_msAcc_RUB_HANDS))
ts_msGyr_Two_Activities = np.concatenate((ts_msGyr_OPEN_DOOR, ts_msGyr_RUB_HANDS))
ts_labels_Two_Activities = np.concatenate((ts_labels_OPEN_DOOR, ts_labels_RUB_HANDS))

np.save("test_MSAccelerometer_OpenDoor_RubHands.npy", ts_msAcc_Two_Activities)
np.save("test_MSGyroscope_OpenDoor_RubHands.npy", ts_msGyr_Two_Activities)
np.save("test_labels_OpenDoor_RubHands.npy", ts_labels_Two_Activities)
