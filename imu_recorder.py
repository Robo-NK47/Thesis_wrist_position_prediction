from __future__ import print_function
import serial as ser
import struct
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import os
import time
from os import listdir
from os.path import isfile, join
from NatNetClient import NatNetClient
from collections import defaultdict
import seaborn as sns
from mpl_toolkits import mplot3d
from tqdm.auto import tqdm
from termcolor import colored
import itertools


def receiveNewFrame(frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged):
    print("Received frame", frameNumber)


# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame(id, position, rotation):
    print("Received frame for rigid body", id)


def receiveRigidBodyList(rigidBodyList, stamp):
    for (ac_id, pos, quat, valid) in rigidBodyList:
        if not valid:
            # skip if rigid body is not valid
            continue
        # print('id: ', ac_id, 'pos:', pos, 'quat:', quat)


def countdown(time_factor):
    # Gives the user a few seconds to prepare for a
    # data recording session
    time.sleep(0.5)
    print(" ")
    print("The recording will begin in:")
    for i in range(0, 3):
        print(3 - i)
        time.sleep(time_factor)
    print(" ")


def get_data_from_arduino():
    while True:
        if ard.inWaiting() > 0:
            current_data = ard.read(4)
            packet = struct.unpack('<f', current_data)[0]
            return packet


def get_data(show_raw_data):
    keys = ['G1', 'A1', 'M1', 'G2', 'A2', 'M2']
    serial_keys = [1000001.0, 1000002.0, 1000003.0, 2000001.0, 2000002.0, 2000003.0]
    all_imu_data = []

    for key in keys:
        current_data_for_sens_type = [key]
        counter = 0
        current_data = 0
        while current_data != serial_keys[keys.index(key)]:
            current_data = get_data_from_arduino()
            if show_raw_data:
                sys.stdout.write(colored("\r" + f"Establishing connection, the current value read from the "
                                                f"controller is: {current_data}", 'green'))
                sys.stdout.flush()
            if counter > 100:
                return None
            counter += 1
        for j in range(3):
            raw = get_data_from_arduino()
            current_data_for_sens_type.append(raw)
        all_imu_data.append(current_data_for_sens_type)

    return all_imu_data


def establish_connection(show_raw_data, baud_rate, COM):
    counter = 0
    plural = ["attempts", "attempt"]
    while True:
        global ard
        try:
            ard = ser.Serial(COM, baud_rate)
        except ser.serialutil.SerialException:
            # print("Connection to Arduino lost, closing script.")
            # sys.exit(0)
            pass

        if get_data(show_raw_data) is not None:
            print("\n\nConnection established, Data record will begin soon.\n")
            if counter == 1:
                return f'\nConnectivity report:\n' \
                       f'{counter} Failed {plural[1]} before establishing good connectivity to controller.\n'
            else:
                return f'\nConnectivity report:\n' \
                       f'{counter} Failed {plural[0]} before establishing good connectivity to controller.\n'
        ard.close()
        counter += 1


def plot_data(data, name, to_save, IMU_type):
    data_types = [('Gyroscope', '[deg/sec]'), ('Accelerometer', '[g]'), ('Magnetic sensor', '[Gauss]')]
    axis = ['x', 'y', 'z']
    x = range(data.shape[1])

    fig, axs = plt.subplots(3, figsize=(20, 10))
    fig.suptitle(f'{IMU_type} IMU measurments')
    data_to_plot = data.reshape(3, 3, -1)

    for i, ax in enumerate(axs.flat):
        ax.set_title(data_types[i][0])
        ax.set(xlabel='Sample', ylabel=data_types[i][1])
        ax.label_outer()
        for j in range(3):
            ax.plot(x, data_to_plot[i][j], label=axis[j])
            ax.legend(loc="upper right")

    dir_path = r'C:\Users\User\Documents\IMU\Raw data\IMU_graphs'
    file_name = name + f' {IMU_type}.png'
    path = os.path.join(dir_path, file_name)
    if to_save:
        fig.savefig(path, dpi=100, bbox_inches='tight')


def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def threeD_graph(obj):
    elbow_position = np.array(obj['elbow_position']).transpose()
    shoulder_position = np.array(obj['shoulder_position']).transpose()
    wrist_position = np.array(obj['wrist_position']).transpose()

    sns.set(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter3D(elbow_position[0], elbow_position[1], elbow_position[2], color='orange', label='Elbow')
    ax.scatter3D(shoulder_position[0], shoulder_position[1], shoulder_position[2], color='red', label='Shoulder')
    ax.scatter3D(wrist_position[0], wrist_position[1], wrist_position[2], color='yellow', label='Wrist')
    ax.legend()
    return [fig, ax]


def count_empy_samples(data):
    keyz = ['elbow', 'shoulder', 'wrist']
    counter_dict = {}.fromkeys(keys, 0)
    percentages = []
    for key in keyz:
        position = data[key + '_position']
        for cell in position:
            if cell == (0.0, 0.0, 0.0):
                counter_dict[key] += 1
        percentages.append((1 - ((len(position) - counter_dict[key]) / len(position))) * 100)

    print('\nFor this recording:')
    for i, key in enumerate(counter_dict):
        print(f'The {key} element is {percentages[i]:.1f}% zeroes.')
    return percentages


take_input = True
cols = [0, 1, 2, 3, 4, 5]
rows = ['D', 'E', 'F', 'G']
cols = [5]
rows = ['A']
user_name = input("What is your name? ")
reconnect = False
############################################################################################
while True:
    try:
        natnet = NatNetClient(rigidBodyListListener=receiveRigidBodyList)
        natnet.run()
        break
    except struct.error:
        pass
connection = establish_connection(True, 57600, 'COM5')
############################################################################################

for grid_cell in itertools.product(rows, cols):
    while True:
        if reconnect:
            while True:
                try:
                    connection = establish_connection(True, 57600, 'COM9')
                    natnet = NatNetClient(rigidBodyListListener=receiveRigidBodyList)
                    natnet.run()
                    break
                except struct.error:
                    pass
        ############################################################################################
        file_path = r'C:\Users\User\Documents\IMU\Raw data\IMU_raw_data'
        ############################################################################################
        data_points_to_record = 720
        print(connection)
        data_points = 1
        to_plot = False
        record = False
        first_step = True
        all_imu_raw_data = []
        times = []
        marker_dict = defaultdict(list)
        keys = ['wrist', 'elbow', 'shoulder', 'matrix']

        wrist_id = 3
        elbow_id = 4
        shoulder_id = 5
        matrix_id = 999

        # This dictionary matches the rigid body id (key) to it's name (value)
        motive_matcher = {wrist_id: 'wrist',
                          elbow_id: 'elbow',
                          shoulder_id: 'shoulder',
                          matrix_id: 'matrix'}

        mistakes_dict = defaultdict(int)
        for key in keys:
            mistakes_dict[key] = 0
        ############################################################################################
        list_of_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        file_id = len(list_of_files)
        text = colored(f'#################################################################################################'
                       f'######', 'white') + colored('\n# 1. Press any key to begin recording, press the yellow button to '
                       'stop the recording.                 #', 'blue') + colored(f"\n# 2. Enter 'e' to exit the recording "
                       f"process (there are currently {file_id} files in the recordings folder)   #", 'blue') + colored('\n'
                       '##################################################################################################'
                       '#####\n', 'white')
        if take_input:
            p = input(text)
            take_input = False

        if p == 'e':
            print(colored("\n***  Thank you, come again.  ***", 'magenta'))
            ard.close()
            sys.exit(0)
        text = colored(f'\n###########################\n'
                       f'#   Go to cell {grid_cell}!  #\n'
                       f'###########################\n', 'red')

        print(text)

        good_data = True
        error_start = time.time()
        temp_time = time.time()
        for i in tqdm(range(300)):
            a = get_data(False)
            delt = time.time() - temp_time
            temp_time = time.time()
            if delt > (1/20.0):
                good_data = False
                break

            # else:
            #     take_input = True
        print("________________________________________________________________________________________ GO!!!!! ")
        if good_data:
            if time.time() - error_start > 1:
                record = True
                for i in tqdm(range(data_points_to_record)):
                    a = get_data(False)
                    if record:
                        if a is None:
                            print('\n\nEnd of recording.\n')
                            record = False
                            break
                        elif isinstance(a, list):
                            single_loop = []
                            for j in range(len(a)):
                                single_loop = single_loop + a[j][1:]

                        data_points += 1
                        times.append(time.time())
                        # print(single_loop[3:6])
                        all_imu_raw_data.append(single_loop)
                        marker_data = natnet.call()
                        for j, cell in enumerate(marker_data):
                            if cell[0] in list(motive_matcher.keys()):
                                key = motive_matcher[cell[0]]
                                data = {'position': cell[1], 'rotation': cell[2]}
                                if data['position'] == (0.0, 0.0, 0.0) and not first_step:
                                    marker_dict[key + '_position'].append(marker_dict[key + '_position'][-1])
                                    marker_dict[key + '_position'].append(marker_dict[key + '_position'][-1])
                                    mistakes_dict[key] += 1
                                else:
                                    marker_dict[key + '_position'].append(data['position'])
                                    marker_dict[key + '_rotation'].append(data['rotation'])
                        if first_step:
                            start = time.time()
                            first_step = False

                # ard.close()
                # natnet.stop()
                plt.close('all')
                print("____________________________________________________________"
                      " Place your hand to the side of your body.")

                for key in keys:
                    if key in list(marker_dict.keys()):
                        del marker_dict[key]

                organized_imu_raw_data = np.array(all_imu_raw_data).transpose()

                percentages = []
                for key in keys:
                    score = (1 - (data_points - mistakes_dict[key]) / data_points) * 100
                    percentages.append(score)

                if max(percentages) > 25:
                    for i, key in enumerate(keys):
                        print(f'The {key} element contains {percentages[i]:.1f}% missed frames.')
                    to_save = False
                    print("\n!!!!!!!!!! Too many missed markers, this recording will not be saved. !!!!!!!!!!\n")
                else:
                    to_save = True

                file_name = f'{file_id} IMU record {len(all_imu_raw_data)} data points cell {grid_cell} id' \
                            f' {time.time()} - {user_name}'

                if to_save:
                    if data_points > 0:
                        wrist_imu_data = organized_imu_raw_data[0:9, 0:data_points_to_record]
                        plot_data(wrist_imu_data, file_name, True, 'Wrist')

                        arm_imu_data = organized_imu_raw_data[9:18, 0:data_points_to_record]
                        plot_data(arm_imu_data, file_name, True, 'Arm')

                    path = os.path.join(r'C:\Users\User\Documents\IMU\Raw data\IMU_raw_data', file_name + '.pkl')
                    save_dict = {'imu_data': organized_imu_raw_data,
                                 'marker_data': marker_dict,
                                 'time_stamps': times}
                    save(path, save_dict)
                    break

                print("############################################################################################\n"
                      "############################################################################################\n\n")
        else:
            ard.close()
            reconnect = True
            take_input = False
            natnet.stop()
            print("########################################################")
            print(colored("<<<<<<<<<<  Arduino out of sync, re-syncing.  >>>>>>>>>>", 'red'))
            print("########################################################\n\n")
        plt.close('all')
        ############################################################################################
sys.exit(0)
