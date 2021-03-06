from __future__ import print_function

import os
import socket
import pickle
from serial import Serial
from serial import serialutil
from serial import SerialException
import struct
import numpy as np
import pickle
import sys
import time
from tqdm.auto import tqdm
from termcolor import colored
from scipy.signal import savgol_filter
import glob
import torch
from data_class_MLP import NN
from data_class_LSTM_CNN import LSTM_CNN


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = Serial(port)
            s.close()
            result.append(port)
        except (OSError, SerialException):
            pass
    return result


def load_model(model_path):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = torch.load(model_path, map_location=torch.device(device))
    model.device = device
    return model


def preform_pca(data, pca_parameters):
    x_shape = data[0].shape
    data = pca_parameters.transform(np.array(data).reshape(-1, x_shape[-1]))
    data = data.reshape(-1, pca_parameters.n_components_)

    return data


def load_file(path):
    with (open(path, "rb")) as openfile:
        try:
            return pickle.load(openfile)
        except EOFError:
            return 'File error.'


def scale_data(data, scalars):
    sample_shape = data.shape
    number_of_features = sample_shape[-1]
    data = data.reshape(-1, number_of_features).transpose()

    all_scaled_xs = []
    for feature, scalar in zip(data, scalars):
        scaled_feature = scalar.transform(feature.reshape(-1, 1))
        all_scaled_xs.append(scaled_feature.reshape(-1))

    all_scaled_xs = np.array(all_scaled_xs).transpose()
    all_scaled_xs = all_scaled_xs.reshape(sample_shape)

    return all_scaled_xs


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
            if abs(packet) > 1e+30:
                establish_connection(False, 57600, '/dev/ttyACM0')
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

    # print(f'Got raw data\n{all_imu_data}\n\n')
    return all_imu_data


def establish_connection(show_raw_data, baud_rate, COM):
    counter = 0
    plural = ["attempts", "attempt"]
    while True:
        global ard
        try:
            ard = Serial(COM, baud_rate)
        except serialutil.SerialException:
            return "Connection to Arduino lost."
            # sys.exit(0)
            pass

        if get_data(show_raw_data) is not None:
            # print("\n\nConnection established, Data record will begin soon.\n")
            if counter == 1:
                return f'\nConnectivity report:\n' \
                       f'{counter} Failed {plural[1]} before establishing good connectivity to controller.\n'
            else:
                return f'\nConnectivity report:\n' \
                       f'{counter} Failed {plural[0]} before establishing good connectivity to controller.\n'
        ard.close()
        counter += 1


def savgol_filtering(data, window_size, polynomial_order):
    return savgol_filter(data, window_size, polynomial_order, axis=0)


def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def get_imu_single_vector():
    single_loop = []
    while len(single_loop) == 0:
        a = get_data(False)
        # print(f'Got a read from the arduino\n {a}\n\n')
        if isinstance(a, list):
            single_loop = []
            for j in range(len(a)):
                single_loop = single_loop + a[j][1:]

    return np.array(single_loop)


def stable_the_ard():
    temp_time = time.time()
    for i in tqdm(range(300)):
        a = get_data(False)
        delt = time.time() - temp_time
        temp_time = time.time()
    if delt > (1 / 20.0):
        return 'bad_data'


def get_predictions():
    available_ports = serial_ports()
    # for i, port in enumerate(available_ports):
    #     print(f'{i} - {port}')
    #
    # port_choice = input('\nPlease enter the number of desired port (should be /dev/ttyACM0): ')

    arduino_connection = establish_connection(True, 57600, available_ports[0])

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 12345))
    s.listen(1)

    client_socket, address = s.accept()
    print(f'Connection from {address} has been established.')
    client_listening = True

    def get_MLP_predictions(data, model):
        # Normalize
        data = scale_data(data, MLP_normalizing_scalars)
        # Scale
        data = scale_data(data, MLP_scaling_scalars)

        prediction = MLP_model(torch.transpose(torch.Tensor(data).float().view(-1, 1), 0,
                                               1).to(model.device)).cpu().detach().numpy()[0]
        return prediction

    def prepare_data_to_LSTM(data, MLP_prediction):
        # Normalize
        _input = scale_data(np.array(data), LSTM_normalizing_scalars)

        # Savgol filter
        _input = savgol_filtering(_input, 59, 6)

        # PCA
        _input = preform_pca(_input, LSTM_pca)

        # Scale data
        _input = scale_data(_input, LSTM_scaling_scalars)

        # Add MLP predictions
        _input = np.concatenate([_input, np.array(MLP_prediction)], axis=1)

        _input = torch.Tensor(_input).to(LSTM_model.device)

        return torch.unsqueeze(_input, 0)

    work_dir = os.getcwd()

    # MLP components #
    MLP_parameters = load_file(os.path.join(work_dir, 'MLP parameters.pkl'))
    MLP_scaling_scalars = MLP_parameters['scaling_scalers']
    MLP_normalizing_scalars = MLP_parameters['normalizing_scalers']
    MLP_model = load_model(os.path.join(work_dir, 'MLP/model - best.pt'))

    # LSTM components #
    LSTM_parameters = load_file(os.path.join(work_dir, 'LSTM parameters.pkl'))
    LSTM_scaling_scalars = LSTM_parameters['scaling_scalers']
    LSTM_normalizing_scalars = LSTM_parameters['normalizing_scalers']
    LSTM_pca = LSTM_parameters['pca']
    LSTM_model = load_model(os.path.join(work_dir, 'LSTM/model - best.pt'))

    time_step_amount = 60
    raw_imu_read = []
    if stable_the_ard() != 'bad_data':
        counter = 0
        while True:
            if len(raw_imu_read) < time_step_amount:
                first_read = True

            if first_read:
                raw_imu_read = []
                MLP_predictions = []
                while len(raw_imu_read) < time_step_amount:
                    single_imu_read = get_imu_single_vector()
                    MLP_predictions.append(get_MLP_predictions(single_imu_read, MLP_model))
                    raw_imu_read.append(single_imu_read)
                first_read = False

            else:
                raw_imu_read.pop(0)
                MLP_predictions.pop(0)

                single_imu_read = get_imu_single_vector()
                raw_imu_read.append(single_imu_read)
                MLP_prediction = get_MLP_predictions(single_imu_read, MLP_model)
                if np.isnan(MLP_prediction).any():
                    first_read = True
                MLP_predictions.append(MLP_prediction)

            if len(raw_imu_read) == time_step_amount and not first_read:
                LSTM_input = prepare_data_to_LSTM(raw_imu_read, MLP_predictions)
                LSTM_prediction = LSTM_model(LSTM_input)[0][0]
                if torch.isnan(LSTM_prediction).any():
                    breakpoint()
                LSTM_prediction = LSTM_prediction.tolist()
                # print(f'{counter} - {LSTM_prediction}\n\n')
            counter += 1
            if counter % 100:
                data = pickle.dumps(LSTM_prediction)
                print(LSTM_prediction)
            try:
                client_socket.send(data)

            except ConnectionResetError:
                client_listening = False
                print('No Client')

            if not client_listening:
                client_socket, address = s.accept()
                print(f'Connection from {address} has been established.')
                client_listening = True


if __name__ == '__main__':
    get_predictions()
    sys.exit(0)
