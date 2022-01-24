import pickle
from pathlib import Path
import os
import pandas as pd
import numpy as np
from data_class_MLP import DataSet, DataSample
from tqdm.auto import tqdm
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import random
from sklearn import preprocessing


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


def marker_data_to_np(data):
    arranged_data = []
    # if len(data) == 8:
    #     del data['matrix_position'], data['matrix_rotation']
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

    print("Augmenting the data:")
    for _ in tqdm(range(number_of_new_datasets)):
        new_imu = np.concatenate([new_imu, make_bias_noised_data(imu)])
        new_marker = np.concatenate([new_marker, marker])
        new_times = np.concatenate([new_times, times])
        new_target_cells = np.concatenate([new_target_cells, target_cells])

    return new_imu, new_marker, new_times, new_target_cells


def remove_outliers(imu, marker, times, target_cells, z):
    imu_entries = np.array(get_filtered_entries(pd.DataFrame(imu.reshape(-1, imu.shape[-1])), z)).reshape(imu.shape[0],
                                                                                                          -1)
    if all(np.equal(marker[0][6:12], np.array([0., 0., 0., 0., 0., 0.]))):
        for _ in range(6):
            marker = np.delete(marker, 6, 1)

    marker_filtered_entries = get_filtered_entries(pd.DataFrame(marker.reshape(-1, marker.shape[-1])), z)

    filtered_entries = []
    for imu_sample, marker_sample in zip(imu_entries, marker_filtered_entries):
        if all(imu_sample) and marker_sample:
            filtered_entries.append(True)
        else:
            filtered_entries.append(False)

    new_imu = []
    new_marker = []
    new_times = []
    new_target_cells = []

    for imu_sample, marker_sample, time_sample, target_cell, entry in zip(imu, marker, times,
                                                                          target_cells, filtered_entries):
        if entry:
            new_imu.append(imu_sample)
            new_marker.append(marker_sample)
            new_times.append(time_sample)
            new_target_cells.append(target_cell)

    return np.array(new_imu), np.array(new_marker), np.array(new_times), np.array(new_target_cells)


def get_filtered_entries(data, z):
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    return (abs_z_scores < z).all(axis=1)


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
    data = data.reshape(-1, pca_parameters.n_components_)

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
    print("Scaling the data:")
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


def number_of_active_features(features):
    counter = 0
    for sensor in features:
        for axis in features[sensor]:
            if features[sensor][axis]:
                counter += 1

    return counter


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

        new_recording = np.array([shoulder_position, shoulder_rotation, elbow_position,
                                  elbow_rotation, wrist_position, wrist_rotation]).reshape(-1, 120).transpose()
        new_data.append(new_recording)

    return np.array(new_data)


def marker_data_relative_to_shoulder(data):
    new_data = []
    for recording in data:
        recording = recording.transpose().reshape(-1, 3, 120)
        wrist_position = recording[4] - recording[0]
        wrist_rotation = recording[5] - recording[1]
        elbow_position = recording[2] - recording[0]
        elbow_rotation = recording[3] - recording[1]

        zeros = np.zeros(wrist_position.shape)
        new_recording = np.array([wrist_position, wrist_rotation, elbow_position,
                                  elbow_rotation, zeros, zeros]).reshape(-1, 120).transpose()
        new_data.append(new_recording)

    return np.array(new_data)


avish_dict = {}.fromkeys(['TRAIN', 'TEST'])
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
num_of_active_imu_features = number_of_active_features(imu_features)

marker_features = {'shoulder_position': {'x': False, 'y': False, 'z': False},
                   'shoulder_rotation': {'x': False, 'y': False, 'z': False},
                   'elbow_position': {'x': False, 'y': False, 'z': False},
                   'elbow_rotation': {'x': False, 'y': False, 'z': False},
                   'wrist_position': {'x': True, 'y': True, 'z': True},
                   'wrist_rotation': {'x': False, 'y': False, 'z': False}}
num_of_active_marker_features = number_of_active_features(marker_features)
#################################################
preprocessing_operations = {'noise': {'generate': False,
                                      'how_many_to_generate': 1},
                            'shuffle_data': False,
                            'scale': True,
                            'normalize': True,
                            'outliers': {'to_remove': True, 'z_score': 3},
                            'PCA': {'to_preform': False, 'pca_percent': 0.99},
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

        imu_data = current_raw_recording['imu_data']
        marker_data = marker_data_to_np(current_raw_recording['marker_data'])
        del current_raw_recording

        # imu_data = arrange_by_selected_features(imu_data, imu_features).to_numpy()
        # marker_data = arrange_by_selected_features(marker_data, marker_features).to_numpy()

        all_imu_type_data.append(imu_data.transpose())
        all_marker_type_data.append(marker_data.transpose())
        all_time_stamps_type_data.append(np.array(list(range(len(imu_data.transpose())))))
        all_cell_type_data.append(return_cell_name(file_name))
        del (imu_data, marker_data)

    all_data[train_or_test]['imu'] = np.array(all_imu_type_data)
    all_data[train_or_test]['marker'] = np.array(all_marker_type_data)
    all_data[train_or_test]['time_stamps'] = np.array(all_time_stamps_type_data)
    all_data[train_or_test]['cells'] = np.array(all_cell_type_data)

    del (all_imu_type_data, all_marker_type_data, all_time_stamps_type_data, all_cell_type_data)

#################################################
for train_or_test in all_data:
    print(f"Pre-processing the {train_or_test} data.")
    imu_data = all_data[train_or_test]['imu']
    time_stamps = all_data[train_or_test]['time_stamps']
    cells = all_data[train_or_test]['cells']
    marker_data = all_data[train_or_test]['marker']

    ## Make marker data relative to matrix ##
    if preprocessing_operations['marker_data_relative_to_matrix']:
        # marker_data = marker_data_relative_to_shoulder(marker_data)
        marker_data = marker_data_relative_to_matrix(marker_data)

    ## Normalize imu data ##
    if preprocessing_operations['normalize']:
        if train_or_test == 'TRAIN':
            normalizing_scalers = get_feature_scalars(imu_data, 'standard')
        imu_data = scale_data(imu_data, normalizing_scalers)

    ## Add noise ##
    if train_or_test == 'TRAIN' and preprocessing_operations['noise']['generate']:
        new_sets = preprocessing_operations['noise']['how_many_to_generate']
        imu_data, marker_data, time_stamps, cells = augment_data(imu_data, marker_data, time_stamps, cells, new_sets)

    ## Split into separate samples ##
    recording_time = imu_data.shape[1]
    temp_cells = []
    for cell in cells:
        for _ in range(recording_time):
            temp_cells.append(cell)

    cells = np.array(temp_cells)
    imu_data = imu_data.reshape(-1, 18)
    marker_data = marker_data.reshape(-1, 18)
    time_stamps = time_stamps.reshape(-1)

    ## Remove outliers ##
    if preprocessing_operations['outliers']['to_remove'] and train_or_test == 'TRAIN':
        sample_amount = len(imu_data)
        z_score = preprocessing_operations['outliers']['z_score']
        imu_data, marker_data, time_stamps, cells = remove_outliers(imu_data, marker_data, time_stamps, cells, z_score)
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
    for imu_sample, marker_sample, cell, time_stamp in zip(imu_data, marker_data, cells, time_stamps):
        sample = DataSample(x=imu_sample,
                            y=marker_sample,
                            target_cell=cell,
                            time_range=(time_stamp, time_stamp))
        dataset.append(sample)

    preprocessing_operations['unique_time_ranges'] = list({}.fromkeys(time_stamps).keys())
    preprocessing_operations['unique_cells'] = list({}.fromkeys(cells).keys())

    all_data[train_or_test] = DataSet(general_set=preprocessing_operations,
                                      imu_settings=imu_features,
                                      marker_settings=marker_features,
                                      data_set=dataset)

    ## Shuffle the datasets ##
    if preprocessing_operations['shuffle_data']:
        random.shuffle(all_data[train_or_test].data_set)

    avish_dict[train_or_test] = marker_data

print(f"\n\nThere are {len(all_data['TRAIN'])} training sample.")
print(f"There are {len(all_data['TEST'])} testing sample.\n\n")

train_save_path = os.path.join(save_path, f'train dataset - MLP.pkl')
save(train_save_path, all_data['TRAIN'])

test_save_path = os.path.join(save_path, f'test dataset - MLP.pkl')
save(test_save_path, all_data['TEST'])

MLP_data_dict = {'scaling_scalers': scaling_scalers,
                 'normalizing_scalers': normalizing_scalers,
                 'z_score': z_score,
                 'pca': pca_parameter}

test_save_path = os.path.join(save_path, f'MLP parameters.pkl')
save(test_save_path, MLP_data_dict)

print("\n\nPreprocessing Done.\n\n")
