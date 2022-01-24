import pickle
import os
from mpl_toolkits import mplot3d
import shutil
import matplotlib.pyplot as plt
from data_class_MLP import DataSet, DataSample
from collections import defaultdict
from tqdm.auto import tqdm
import Dataset_maker_For_MLP
import numpy as np


def load_file(path):
    with (open(path, "rb")) as openfile:
        try:
            return pickle.load(openfile)
        except EOFError:
            return 'File error.'


def make_a_threeD_graph(ax, trajectory, title, imu_or_marker):
    print(f'\nMaking a 3D {imu_or_marker} {title} graph:')
    traj = []

    for sample in tqdm(trajectory):
        x = sample['x']
        y = sample['y']
        z = sample['z']
        traj.append(np.array([x, y, z]))

    traj = np.array(traj)
    ax.scatter3D(traj[:, 0], traj[:, 1], traj[:, 2])
    # lims = [-1, 1]
    # ax.set_xlim(lims)
    # ax.set_ylim(lims)
    # ax.set_zlim(lims)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.close('all')
    return ax


def make_neat_data(data, key, keys, imu_or_marker):
    neat_data = []
    for sample in data.data_set:
        if imu_or_marker == 'imu' or imu_or_marker == 'PCA':
            sam = data.fetch_item(sample.x, data.imu_settings, data.preprocessing_settings)

        if imu_or_marker == 'marker':
            sam = data.fetch_item(sample.y, data.marker_settings, data.preprocessing_settings)

        neat_data.append({'x': sam[keys[key]['x']],
                          'y': sam[keys[key]['y']],
                          'z': sam[keys[key]['z']]})

    return neat_data


def get_data_keys(settings):
    keyz = defaultdict(defaultdict, {k: None for k in list(settings.keys())})
    counter = 0
    for sensor in settings:
        axis_dict = defaultdict(int, {k: int for k in list(settings[sensor].keys())})
        for i, axis in enumerate(settings[sensor]):
            if settings[sensor][axis]:
                axis_dict[axis] = counter
                counter += 1
            else:
                print(f'Deleting {sensor}')
                del keyz[sensor]
                break
        if settings[sensor][axis]:
            keyz[sensor] = axis_dict

    return keyz


def graph_maker(keys, unit_type, train_data, test_data, path, text):
    for key in keys:
        if unit_type == 'imu':
            sensor_type = key[0:key.find('_')]
            measurement_unit = measurement_unit_dict[sensor_type]
        if unit_type == 'marker':
            measurement_unit = '[mm]'
        if unit_type == 'PCA':
            measurement_unit = '[PCA projection]'

        fig = plt.figure(figsize=(30, 15))
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2 = fig.add_subplot(1, 2, 1, projection='3d')

        ax2 = make_a_threeD_graph(ax2, make_neat_data(train_data, key, keys, unit_type), 'Train data', unit_type)
        ax1 = make_a_threeD_graph(ax1, make_neat_data(test_data, key, keys, unit_type), 'Test data', unit_type)

        fig.suptitle(f"All {key} data\n Measured in: {measurement_unit}")
        fig_save_path = os.path.join(path, f'Raw data 3D {unit_type} graph - {key}.png')

        fig.text(0.1, 0.1, text, horizontalalignment='left', verticalalignment='center', fontsize=13)
        fig.savefig(fig_save_path)
        plt.close('all')


operating_system = 'linux'

if operating_system == 'windows':
    data_dir = r'C:\Users\User\PycharmProjects\IMU\Raw data\datasets\Preprocessed'

if operating_system == 'linux':
    data_dir = r'/home/nadavk/IMU/Raw data/datasets/Preprocessed/'


save_path = os.path.join(data_dir, 'figures')

for file_name in os.listdir(save_path):
    file_path = os.path.join(save_path, file_name)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s.' % (file_path, e))


measurement_unit_dict = {'Gyroscope': '[rad/sec]', 'Accelerometer': '[g]', 'Magnetometer': '[Gauss]'}

datas = {'Test': load_file(os.path.join(data_dir, 'test dataset - MLP.pkl')),
         'Train': load_file(os.path.join(data_dir, 'train dataset - MLP.pkl'))}

dataset_settings = datas['Train'].preprocessing_settings

imu_settings = datas['Train'].imu_settings
marker_settings = datas['Train'].marker_settings

fig_text = 'Preprocessing methods:'
for key in dataset_settings:
    fig_text += f'\n{key}: {dataset_settings[key]}'

imu_keys = get_data_keys(imu_settings)
marker_keys = get_data_keys(marker_settings)

train_data = datas['Train']
test_data = datas['Test']

if not dataset_settings['PCA']['to_preform']:
    graph_maker(imu_keys, 'imu', train_data, test_data, save_path, fig_text)
else:
    keys = {}.fromkeys(list(range(int(train_data[0][0].shape[-1] / 3))))
    for key in keys:
        keys[key] = {'x': True, 'y': True, 'z': True}
    graph_maker(keys, 'PCA', train_data, test_data, save_path, fig_text)
graph_maker(marker_keys, 'marker', train_data, test_data, save_path, fig_text)

print("Done generating dataset graphs.\n\n")
# sys.exit(0)
