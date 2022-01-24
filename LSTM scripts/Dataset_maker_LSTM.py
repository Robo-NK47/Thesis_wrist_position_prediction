import pickle
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from data_class_LSTM_CNN import DataSet, DataSample
from tqdm.auto import tqdm
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import random
from sklearn import preprocessing
import torch


def load(path):
    with (open(path, "rb")) as openfile:
        try:
            return pickle.load(openfile)
        except EOFError:
            return 'File error.'


def save(pat, data):
    with open(pat, 'wb') as f:
        pickle.dump(data, f)


def return_ordered_list_of_files(train_path, test_path):
    train_files = sorted(Path(train_path).iterdir(), key=os.path.getmtime)
    test_files = sorted(Path(test_path).iterdir(), key=os.path.getmtime)

    return {'TRAIN': train_files, 'TEST': test_files}


def arrange_by_selected_features(data, features):
    # Input is a np array of raw data, features is a dictionary of feature state
    # Output is a pandas DataFrame
    chosen_features_dict = {}
    feature_number = 0
    for feature in features:
        for axis in features[feature]:
            if features[feature][axis]:
                chosen_features_dict[(feature, axis)] = data[feature_number]
            feature_number += 1
    return pd.DataFrame.from_dict(chosen_features_dict)


def arrange_by_selected_feature(data, features):
    # Input is a np array of raw data, features is a dictionary of feature state
    # Output is a pandas DataFrame
    chosen_features_dict = {}
    feature_number = 0
    data_shape = data.shape
    data = data.transpose().reshape(data_shape[-1], -1)

    for feature in features:
        for axis in features[feature]:
            if features[feature][axis]:
                chosen_features_dict[(feature, axis)] = data[feature_number]
            feature_number += 1

    return pd.DataFrame.from_dict(chosen_features_dict)


def marker_data_to_np(data):
    arranged_data = []
    for key in data:
        x = []
        y = []
        z = []
        for sample in data[key]:
            x.append(sample[0])
            y.append(sample[1])
            z.append(sample[2])
        arranged_data.append(np.array(x))
        arranged_data.append(np.array(y))
        arranged_data.append(np.array(z))

    return np.array(arranged_data)


def return_cell_name(name):
    name = name.name
    name = name[name.find('('): name.find('(') + 8]
    name = name.replace("'", "")
    return name


def make_bias_noised_data(data):
    noised_data = []
    for sample in data:
        noised_sample = []

        means = get_data_feature_means(sample)
        stds = get_data_feature_stds(sample)
        for feature, mean, std in zip(sample.transpose(), means, stds):
            noise = np.random.normal(mean, np.abs(std), 1)
            noised_sample.append(feature + noise)

        noised_data.append(np.array(noised_sample).transpose())

    return np.array(noised_data)


def get_data_feature_means(data):
    return np.mean(data.reshape(-1, data.shape[-1]).transpose(), axis=1)


def get_data_feature_stds(data):
    return np.mean(data.reshape(-1, data.shape[-1]).transpose(), axis=1)


def augment_data(imu, marker, times, target_cells, number_of_new_datasets):
    new_imu = np.array(imu)
    new_marker = np.array(marker)
    new_times = np.array(times)
    new_target_cells = np.array(target_cells)

    if number_of_new_datasets == 0:
        return make_bias_noised_data(imu), new_marker, new_times, new_target_cells

    print("Augmenting the data:")
    for _ in tqdm(range(number_of_new_datasets)):
        new_imu = np.concatenate([new_imu, make_bias_noised_data(imu)])
        new_marker = np.concatenate([new_marker, marker])
        new_times = np.concatenate([new_times, times])
        new_target_cells = np.concatenate([new_target_cells, target_cells])

    return new_imu, new_marker, new_times, new_target_cells


def moving_time_window(imu, marker, times, target_cells, window_step, x_steps, y_steps_from_last, option):
    new_imu_data = []
    new_marker_data = []
    new_times_data = []
    new_cells_data = []

    for imu_recording, marker_recording, time_range, cell in zip(imu, marker, times, target_cells):
        counter = 0
        if option == 1:
            while (x_steps + counter) < len(imu_recording):
                new_imu_data.append(imu_recording[counter: x_steps + counter])
                new_marker_data.append(marker_recording[-1].reshape(1, -1))
                new_times_data.append(time_range[counter: x_steps + counter])
                new_cells_data.append(cell)
                counter += window_step

        if option == 2:
            while (2 * x_steps + counter) < len(imu_recording):
                new_imu_data.append(imu_recording[counter: x_steps + counter])
                new_marker_data.append(marker_recording[2 * x_steps + counter - y_steps_from_last: 2 * x_steps + counter])
                new_times_data.append(time_range[counter: x_steps + counter])
                new_cells_data.append(cell)
                counter += window_step

        if option == 3:
            while (x_steps + counter) < len(imu_recording):
                new_imu_data.append(imu_recording[counter: x_steps + counter])
                new_marker_data.append(marker_recording[x_steps + counter - y_steps_from_last: x_steps + counter])
                new_times_data.append(time_range[counter: x_steps + counter])
                new_cells_data.append(cell)
                counter += window_step

    return np.array(new_imu_data), np.array(new_marker_data), np.array(new_times_data), np.array(new_cells_data)


def remove_outliers(imu, marker, times, target_cells, z, s_terms):
    new_imu = []
    new_marker = []
    new_times = []
    new_target_cells = []
    new_start_terms = []

    imu_filtered_entries = np.array(get_filtered_entries(pd.DataFrame(imu.reshape(-1, imu.shape[-1])), z))
    marker_filtered_entries = get_filtered_entries(pd.DataFrame(marker.reshape(-1, marker.shape[-1])), z)
    if imu_filtered_entries.shape[-1] == 0 or marker_filtered_entries.shape[-1] == 0:
        return np.array(new_imu), np.array(new_marker), np.array(new_times), np.array(new_target_cells), \
           np.array(new_start_terms)

    imu_entries = imu_filtered_entries.reshape(imu.shape[0], -1)
    filtered_entries = []
    for imu_sample, marker_sample in zip(imu_entries, marker_filtered_entries):
        if all(imu_sample) and marker_sample:
            filtered_entries.append(True)
        else:
            filtered_entries.append(False)

    for imu_sample, marker_sample, time_sample, target_cell, s_term, entry in zip(imu, marker, times,
                                                                                  target_cells, s_terms,
                                                                                  filtered_entries):
        if entry:
            new_imu.append(imu_sample)
            new_marker.append(marker_sample)
            new_times.append(time_sample)
            new_target_cells.append(target_cell)
            new_start_terms.append(s_term)

    return np.array(new_imu), np.array(new_marker), np.array(new_times), np.array(new_target_cells), \
           np.array(new_start_terms)


def get_filtered_entries(data, z):
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    return (abs_z_scores < z).all(axis=1)


def savgol_filtering(data, window_size, polynomial_order):
    filtered_data = []
    print('Filtering data:')
    for sample in tqdm(data):
        a_sample = []
        for feature in sample.transpose():
            a_sample.append(savgol_filter(feature, window_size, polynomial_order))
        filtered_data.append(np.array(a_sample).transpose())

    return np.array(filtered_data)


def get_pca_parameters(data, pca_percent):
    pca = PCA(pca_percent)
    x_features = data[0].shape[-1]
    data = data.reshape(-1, x_features)
    return pca.fit(data)


def preform_pca(data, pca_parameters):
    x_shape = data[0].shape
    print(f'\nThe PCA process reduced the dimensionality from {x_shape[-1]} features to '
          f'{pca_parameters.n_components_} features.')
    data = pca_parameters.transform(np.array(data).reshape(-1, x_shape[-1]))
    data = data.reshape(-1, x_shape[0], pca_parameters.n_components_)

    return data


def get_feature_scalars(data, scaler_type):
    print("Getting data scalers.")
    sample_shape = data[0].shape
    number_of_features = sample_shape[-1]

    data = data.reshape(-1, number_of_features).transpose()
    all_scalers = []
    for feature in data:
        if scaler_type == 'min_max':
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        if scaler_type == 'standard':
            scaler = preprocessing.StandardScaler()

        all_scalers.append(scaler.fit(feature.reshape(-1, 1)))

    return all_scalers


def scale_data(data, scalars):
    print("Scaling the data.")
    sample_shape = data[0].shape
    number_of_features = sample_shape[-1]
    data = data.reshape(-1, number_of_features).transpose()

    all_scaled_xs = []
    for feature, scalar in zip(data, scalars):
        scaled_feature = scalar.transform(feature.reshape(-1, 1))
        all_scaled_xs.append(scaled_feature.reshape(-1))

    all_scaled_xs = np.array(all_scaled_xs).transpose()
    all_scaled_xs = all_scaled_xs.reshape(-1, sample_shape[0], sample_shape[1])

    return all_scaled_xs


def add_start_terms(data, model_path, time_steps):
    # data = data[0:10]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print('Adding start terms into the IMU data:')
    new_data = []
    model = torch.load(model_path, map_location=torch.device(device))
    model.device = device
    model.network_layers.to(device)
    for sample, time_step in tqdm(zip(data, time_steps), total=len(data)):
        model_outputs = torch.Tensor([]).to(device)
        for step, time in zip(sample, time_step):
            # 0:3 - Gyroscope_wrist
            # 3:6 - Accelerometer_wrist
            # 6:9 - Magnetometer_wrist
            # 9:12 - Gyroscope_arm
            # 12:15 - Accelerometer_arm
            # 15:18 - Magnetometer_arm
            model_output = torch.unsqueeze(model(torch.Tensor(step).float().to(device)), dim=0)
            model_outputs = torch.cat([model_outputs, model_output], dim=0)
        new_data.append(model_outputs.cpu().detach().numpy().reshape(-1, 3))

    return np.array(new_data)


def marker_data_relative_to_matrix(data):
    new_data = []
    for recording in data:
        recording = recording.transpose().reshape(-1, 3, 120)
        wrist_position = recording[4] - recording[6]
        wrist_rotation = recording[5] - recording[7]
        elbow_position = recording[2] - recording[6]
        elbow_rotation = recording[3] - recording[7]
        shoulder_position = recording[0] - recording[6]
        shoulder_rotation = recording[1] - recording[7]

        new_recording = np.array([shoulder_position, shoulder_rotation,
                                  elbow_position, elbow_rotation,
                                  wrist_position, wrist_rotation]).reshape(-1, 120).transpose()
        new_data.append(new_recording)

    return np.array(new_data)


banned_cells = ['(T, 6)']
debug_mode = False
operating_system = 'linux'
if operating_system == 'windows':
    raw_data_path = r'C:\Users\User\PycharmProjects\IMU\Raw data\IMU_raw_data'
    train_raw_data_path = r'C:\Users\User\PycharmProjects\IMU\Raw data\IMU_raw_data\TRAIN'
    test_raw_data_path = r'C:\Users\User\PycharmProjects\IMU\Raw data\IMU_raw_data\TEST'
    save_path = r'C:\Users\User\PycharmProjects\IMU\Raw data\datasets\Preprocessed'
if operating_system == 'linux':
    raw_data_path = r'/home/nadavk/IMU/Raw data/IMU_raw_data/'
    train_raw_data_path = r'/home/nadavk/IMU/Raw data/IMU_raw_data/TRAIN/'
    test_raw_data_path = r'/home/nadavk/IMU/Raw data/IMU_raw_data/TEST/'
    save_path = r'/home/nadavk/IMU/Raw data/datasets/Preprocessed/'
#################################################
imu_features = {'Gyroscope_wrist': {'x': True, 'y': True, 'z': True},
                'Accelerometer_wrist': {'x': True, 'y': True, 'z': True},
                'Magnetometer_wrist': {'x': True, 'y': True, 'z': True},
                'Gyroscope_arm': {'x': True, 'y': True, 'z': True},
                'Accelerometer_arm': {'x': True, 'y': True, 'z': True},
                'Magnetometer_arm': {'x': True, 'y': True, 'z': True}}

marker_features = {'shoulder_position': {'x': False, 'y': False, 'z': False},
                   'shoulder_rotation': {'x': False, 'y': False, 'z': False},
                   'elbow_position': {'x': False, 'y': False, 'z': False},
                   'elbow_rotation': {'x': False, 'y': False, 'z': False},
                   'wrist_position': {'x': True, 'y': True, 'z': True},
                   'wrist_rotation': {'x': False, 'y': False, 'z': False}}
#################################################
MLP_features_path = os.path.join(r'/home/nadavk/IMU/Best MLP/', 'MLP parameters.pkl')
MLP_features = load(MLP_features_path)
#################################################
x_time_steps = 60
y_time_steps_from_last = 1

if x_time_steps % 2 == 0:
    savgol_window_size = x_time_steps - 1
else:
    savgol_window_size = x_time_steps - 2
#################################################
preprocessing_operations = {'noise': {'generate': False,
                                      'how_many_to_generate': 1},
                            'shuffle_data': True,
                            'scale': True,
                            'normalize': True,
                            'moving_time_window': {'to_increase': True,
                                                   'window_step': 1},
                            'sample_time_steps': x_time_steps,
                            'savgol_filter': {'to_filter': True,
                                              'window_size': savgol_window_size,
                                              'polinomial_order': 6},
                            'outliers': {'to_remove': True, 'z_score': 3},
                            'PCA': {'to_preform': True, 'pca_percent': 0.99},
                            'add_start_terms': True,
                            'marker_data_relative_to_matrix': True}
#################################################
###            Raw data collection            ###
file_dict = return_ordered_list_of_files(train_raw_data_path, test_raw_data_path)

all_data = {'TRAIN': {'imu': np.array([]), 'marker': np.array([]), 'time_stamps': np.array([]), 'cells': np.array([])},
            'TEST': {'imu': np.array([]), 'marker': np.array([]), 'time_stamps': np.array([]), 'cells': np.array([])}}

for train_or_test in file_dict:
    all_imu_type_data = []
    all_marker_type_data = []
    all_time_stamps_type_data = []
    all_cell_type_data = []
    print(f'Collecting all {train_or_test} raw data.')
    for file_name in tqdm(file_dict[train_or_test]):
        file_path = os.path.join(raw_data_path, train_or_test, file_name.name)
        current_raw_recording = load(file_path)

        raw_imu_data = current_raw_recording['imu_data']
        raw_marker_data = current_raw_recording['marker_data']
        del current_raw_recording

        imu_feat = {'Gyroscope_wrist': {'x': True, 'y': True, 'z': True},
                    'Accelerometer_wrist': {'x': True, 'y': True, 'z': True},
                    'Magnetometer_wrist': {'x': True, 'y': True, 'z': True},
                    'Gyroscope_arm': {'x': True, 'y': True, 'z': True},
                    'Accelerometer_arm': {'x': True, 'y': True, 'z': True},
                    'Magnetometer_arm': {'x': True, 'y': True, 'z': True}}

        imu_data = arrange_by_selected_features(raw_imu_data, imu_feat).to_numpy()
        # marker_data = arrange_by_selected_features(marker_data_to_np(raw_marker_data), marker_features).to_numpy()
        marker_data = marker_data_to_np(raw_marker_data).transpose()
        del raw_imu_data, raw_marker_data
        all_imu_type_data.append(imu_data)
        all_marker_type_data.append(marker_data)
        all_time_stamps_type_data.append(np.array(list(range(len(imu_data)))))
        all_cell_type_data.append(return_cell_name(file_name))
        del (imu_data, marker_data)

    all_data[train_or_test]['imu'] = np.array(all_imu_type_data)
    all_data[train_or_test]['marker'] = np.array(all_marker_type_data)
    all_data[train_or_test]['time_stamps'] = np.array(all_time_stamps_type_data)
    all_data[train_or_test]['cells'] = np.array(all_cell_type_data)

    del (all_imu_type_data, all_marker_type_data, all_time_stamps_type_data, all_cell_type_data)

#################################################
for train_or_test in all_data:
    print(f"Preprocessing the {train_or_test} data.")
    imu_data = all_data[train_or_test]['imu']
    marker_data = all_data[train_or_test]['marker']
    time_stamps = all_data[train_or_test]['time_stamps']
    cells = all_data[train_or_test]['cells']

    ## Make marker data relative to matrix ##
    if preprocessing_operations['marker_data_relative_to_matrix']:
        # marker_data = marker_data_relative_to_shoulder(marker_data)
        marker_data = marker_data_relative_to_matrix(marker_data)

    ## Apply moving time window ##
    if preprocessing_operations['moving_time_window']['to_increase']:
        marker_options = {1: 'Model output is the last point of the recording session',
                          2: 'Model output is the last point of the next cell',
                          3: 'Model output is the last point of the same cell'}
        option = 1
        preprocessing_operations['Marker type'] = (option, marker_options[option])
        step = preprocessing_operations['moving_time_window']['window_step']
        imu_data, marker_data, time_stamps, cells = moving_time_window(imu_data, marker_data, time_stamps, cells, step,
                                                                       x_time_steps, y_time_steps_from_last, option)

    MLP_imu_data = scale_data(imu_data, MLP_features['normalizing_scalers'])

    ## Normalize imu data ##
    if preprocessing_operations['normalize']:
        if train_or_test == 'TRAIN':
            normalizing_scalers = get_feature_scalars(imu_data, 'standard')
        imu_data = scale_data(imu_data, normalizing_scalers)

    ## Add noise ##
    if train_or_test == 'TRAIN' and preprocessing_operations['noise']['generate']:
        new_sets = preprocessing_operations['noise']['how_many_to_generate']
        imu_data, marker_data, time_stamps, cells = augment_data(imu_data, marker_data, time_stamps, cells, new_sets)

    imu_shape = imu_data.shape
    imu_data = arrange_by_selected_feature(imu_data, imu_features).to_numpy()
    imu_data = imu_data.reshape(imu_shape[0], imu_shape[1], -1)

    ## Add start terms from another ANN ##
    if preprocessing_operations['add_start_terms']:
        MLP_imu_data = scale_data(MLP_imu_data, MLP_features['scaling_scalers'])
        model_dir = r'/home/nadavk/IMU/Best MLP/'
        model_pathway = os.path.join(model_dir, 'model - best.pt')
        start_terms = add_start_terms(MLP_imu_data, model_pathway, time_stamps)
    else:
        start_terms = np.zeros((imu_shape[0], imu_shape[1], 3))

    if imu_data.shape[-1] != 0:
        ## Filter the data ##
        if preprocessing_operations['savgol_filter']['to_filter']:
            window_size = preprocessing_operations['savgol_filter']['window_size']
            polynomial_ord = preprocessing_operations['savgol_filter']['polinomial_order']
            imu_data = savgol_filtering(imu_data, window_size, polynomial_ord)

        ## Remove outliers ##
        if preprocessing_operations['outliers']['to_remove'] and train_or_test == 'TRAIN':
            z_score = preprocessing_operations['outliers']['z_score']
            sample_amount = len(imu_data)
            imu_data, marker_data, time_stamps, cells, start_terms = remove_outliers(imu_data, marker_data, time_stamps,
                                                                                     cells, z_score, start_terms)
            new_sample_amount = len(imu_data)
            print(f'The outliers removal process took off {sample_amount - new_sample_amount} samples.\n'
                  f'There are currently {new_sample_amount} sample in the {train_or_test} dataset.\n')

        ## Preform PCA ##
        if train_or_test == 'TRAIN':
            pca_parameter = get_pca_parameters(imu_data, preprocessing_operations['PCA']['pca_percent'])
        if preprocessing_operations['PCA']['to_preform']:
            imu_data = preform_pca(imu_data, pca_parameter)

        ## Scale imu data ##
        if preprocessing_operations['scale']:
            if train_or_test == 'TRAIN':
                scaling_scalers = get_feature_scalars(imu_data, 'min_max')
            imu_data = scale_data(imu_data, scaling_scalers)

        ## Transform to special classes ##
        dataset = []
        if preprocessing_operations['add_start_terms']:
            temp_imu_data = []
            for imu_sample, start_term in zip(imu_data, start_terms):
                temp_imu_data.append(np.concatenate([imu_sample, start_term], axis=1))
            imu_data = np.array(temp_imu_data)

    else:
        ## Transform to special classes ##
        dataset = []
        if preprocessing_operations['add_start_terms']:
            temp_imu_data = []
            for start_term in start_terms:
                temp_imu_data.append(start_term)
            imu_data = np.array(temp_imu_data)
        else:
            print("Dataset is empty, closing script.")
            sys.exit(1)
    final_time_stamps = []
    preprocessing_operations['unique_cells'] = sorted(list({}.fromkeys(cells).keys()))
    for imu_sample, marker_sample, cell, time_stamp in zip(imu_data, marker_data, cells, time_stamps):
        sample = DataSample(x=imu_sample,
                            y=marker_sample,
                            target_cell=cell,
                            time_range=(time_stamp[0], time_stamp[-1]))
        if cell not in banned_cells:
            final_time_stamps.append((time_stamp[0], time_stamp[-1]))
            dataset.append(sample)

    preprocessing_operations['unique_time_ranges'] = sorted(list({}.fromkeys(final_time_stamps).keys()))

    all_data[train_or_test] = DataSet(general_set=preprocessing_operations,
                                      imu_settings=imu_features,
                                      marker_settings=marker_features,
                                      data_set=dataset)

    ## Shuffle the datasets ##
    if preprocessing_operations['shuffle_data']:
        random.shuffle(all_data[train_or_test].data_set)

for train_or_test in all_data:
    if preprocessing_operations['add_start_terms']:
        all_data[train_or_test].imu_settings['model_output'] = {'x': True, 'y': True, 'z': True}

train_save_path = os.path.join(save_path, f'train dataset - LSTM.pkl')
save(train_save_path, all_data['TRAIN'])

test_save_path = os.path.join(save_path, f'test dataset - LSTM.pkl')
save(test_save_path, all_data['TEST'])

if imu_data.shape[-1] == 3 and preprocessing_operations['add_start_terms']:
    ## This case hanldes a datasets which relies only on the MLP output
    pca_parameter = ['Empty - takes input only from MLP.']
    z_score = ['Empty - takes input only from MLP.']
    scaling_scalers = ['Empty - takes input only from MLP.']
    normalizing_scalers = ['Empty - takes input only from MLP.']

LSTM_data_dict = {'scaling_scalers': scaling_scalers,
                 'normalizing_scalers': normalizing_scalers,
                 'z_score': z_score,
                 'pca': pca_parameter}

test_save_path = os.path.join(save_path, f'LSTM parameters.pkl')
save(test_save_path, LSTM_data_dict)

print("\n\nPreprocessing Done.\n\n")
