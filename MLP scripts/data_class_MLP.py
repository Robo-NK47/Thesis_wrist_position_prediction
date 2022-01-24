import torch
from pathlib import Path
import numpy as np
from torch import nn


def get_recording_cell(file_nam):
    # receives a path object, returns the file call target as STR
    file_nam = file_nam.name
    file_nam.replace(r"\r", "")
    letters = 'xxxx'
    position = -1
    while letters != ' id ':
        letters = file_nam[position - len(letters):position]
        position -= 1
    cell_nam = file_nam[position - 11: position]
    return cell_nam[0] + cell_nam[2:3] + cell_nam[4:8]


class DataSample:
    def __init__(self, x, y, target_cell, time_range):
        self.x = x  # a numpy array of the IMU data
        self.y = y  # a numpy array of the marker data
        self.target_cell = target_cell
        self.time_range = time_range

    def __len__(self):
        return len(self.x)


class DataSet(torch.utils.data.Dataset):
    def __init__(self, general_set: dict(), imu_settings: dict(),
                 marker_settings: dict(), data_set: list(), transform=None):
        self.data_set = data_set
        self.excluded_samples = []
        self.sample_amount = len(self.data_set)
        self.preprocessing_settings = general_set
        self.imu_settings = imu_settings
        self.marker_settings = marker_settings
        self.sample_shape = self.shape(self.data_set[0].x, self.data_set[0].y)
        self.transform = transform
        self.time_tag_x = False
        self.all_time_ranges = general_set['unique_time_ranges']
        self.all_cells = general_set['unique_cells']
        self.allowed_times = []
        self.allowed_cells = []

    def __len__(self):
        'Denotes the total number of samples'
        return self.sample_amount

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        'Generates one sample of data'
        if not self.preprocessing_settings['PCA']['to_preform']:
            x = self.fetch_item(self.data_set[idx].x, self.imu_settings, self.preprocessing_settings)

        else:
            x = self.data_set[idx].x

        if self.time_tag_x:
            time_tag = self.data_set[idx].time_range
            x = self.add_time_steps_to_sample(x, time_tag)

        y = self.fetch_item(self.data_set[idx].y, self.marker_settings, self.preprocessing_settings)

        if self.transform:
            x = self.transform(x)
        return x, y, idx

    def get_dataset_name_list(self):
        names = []
        for sample in self.data_set:
            names.append({'target_cell': sample.target_cell,
                          'time_range': sample.time_range})
        return names

    def add_time_steps_to_sample(self, sample, time_range):
        time_tags = (np.array(range(time_range[0], time_range[1] + 1)) / 120).reshape(-1, 1)
        return np.concatenate([sample, time_tags[0]], axis=0)

    def enable_time_step(self):
        self.time_tag_x = True
        if self.time_tag_x:
            self.sample_shape['x'][0] = self.sample_shape['x'][0] + 1

    def fetch_item(self, data, data_settings, preprocessing_settings):
        return_item = []
        data = data.transpose()
        counter = 0
        if preprocessing_settings['PCA']['to_preform'] and ('Gyroscope_wrist' in data_settings):
            return data
        for sensor in data_settings:
            for axis in data_settings[sensor]:
                if data_settings[sensor][axis]:
                    return_item.append(data[counter])
                counter += 1

        return np.array(return_item).transpose()

    def shape(self, x, y):
        return {'x': list(self.fetch_item(x, self.imu_settings, self.preprocessing_settings).shape),
                'y': list(self.fetch_item(y, self.marker_settings, self.preprocessing_settings).shape)}

    def allowed_samples(self, current_error, threshold, current_samples):
        all_cells = self.all_cells
        all_time_ranges = self.all_time_ranges
        addition = 10
        if current_error < threshold:
            if isinstance(current_samples[0], np.int64):
                if len(current_samples) + addition < len(all_time_ranges):
                    self.allowed_times = all_time_ranges[0:len(current_samples) + addition]
                else:
                    self.allowed_times = all_time_ranges
            if isinstance(current_samples[0], str):
                if len(current_samples) + addition < len(all_cells):
                    self.allowed_cells = all_cells[0:len(current_samples) + addition]
                else:
                    self.allowed_cells = all_cells

            self.exclude_samples()

    def exclude_samples(self):
        samples_to_dataset = []
        samples_to_exclude = []

        for sample in self.data_set:
            time_range_condition = sample.time_range[0] in self.allowed_times
            cell_condition = sample.target_cell in self.allowed_cells
            if time_range_condition and cell_condition:
                samples_to_dataset.append(sample)
            else:
                samples_to_exclude.append(sample)

        for sample in self.excluded_samples:
            time_range_condition = sample.time_range[0] in self.allowed_times
            cell_condition = sample.target_cell in self.allowed_cells
            if time_range_condition and cell_condition:
                samples_to_dataset.append(sample)
            else:
                samples_to_exclude.append(sample)

        self.data_set = samples_to_dataset
        self.excluded_samples = samples_to_exclude


class ToTensor(object):
    def __call__(self, x):
        dtype = torch.FloatTensor
        return torch.from_numpy(x).type(dtype)


class NN(nn.Module):
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    def __init__(self, trial, num_of_input_features, num_of_output_features, input_time_steps, output_time_steps,
                 batch_size, device, det_or_prob):
        super(NN, self).__init__()
        self.trial = trial
        self.num_of_FC_layers = self.trial.suggest_int("num_of_FC_layers", 2, 5)
        self.input_feature_size = num_of_input_features
        self.det_or_prob = det_or_prob
        if self.det_or_prob == 'deterministic':
            self.output_feature_size = num_of_output_features
        else:
            self.output_feature_size = num_of_output_features * 2
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.batch_size = batch_size
        self.device = device
        self.network = 0

        self.network_layers = self.make_a_network()

    def make_a_network(self):
        net = []
        input_size = self.input_feature_size
        all_layers = []
        layer_types = ['Linear', 'Activation', 'Dropout', 'BatchNorm']
        for layer_num in range(self.num_of_FC_layers):
            for layer_type in layer_types:
                input_s = input_size
                layer = self.layer_chooser(layer_type, input_size, layer_num)
                input_size = layer['output_size']
                if layer['layer'] is not None:
                    layer = layer['layer']
                    if layer_type == 'Linear':
                        to_skip = self.trial.suggest_int("skip_connection_{}".format(len(all_layers)), 0, 1)
                    else:
                        to_skip = 0

                    if to_skip:
                        faulty = torch.zeros(input_s)
                        faulty = self.skip_connection(faulty, layer)
                        input_size = faulty.shape[0]

                    layer_dict = {'layer_type': layer_type,
                                  'layer_number': len(all_layers),
                                  'input_size': input_s,
                                  'output_size': input_size,
                                  'skip_connection': to_skip}
                    all_layers.append(layer_dict)
                    net.append(layer)

        last = nn.Linear(input_size, self.output_feature_size)
        last_layer = {'layer_type': 'Linear',
                      'layer_number': len(all_layers) + 1,
                      'input_size': input_size,
                      'output_size': (self.batch_size, self.output_time_steps, self.output_feature_size),
                      'skip_connection': 0}
        all_layers.append(last_layer)
        net.append(last)

        self.network = all_layers
        return nn.Sequential(*net)

    def layer_chooser(self, layer_type, input_size, layer_num):
        if layer_type == 'Linear':
            layer = self.FC_layer(layer_num, input_size)

        if layer_type == 'Activation':
            layer = self.activation_function(layer_num, input_size)

        if layer_type == 'Dropout':
            layer = self.dropout(layer_num, input_size)

        if layer_type == 'BatchNorm':
            layer = self.layer_norm(layer_num, input_size)
        # print(layer)
        return layer

    def layer_output_finder(self, layer, input_size):
        layer_output = layer(torch.zeros(self.batch_size, input_size))
        if len(layer_output.shape) > 1:
            layer_output = layer_output[-1]

        return layer_output.shape[-1]

    def FC_layer(self, layer_num, input_dim):
        out_features = self.trial.suggest_int("n_FC_units_{}".format(layer_num), 6, 10)
        out_features = 2 ** out_features
        return {'layer': nn.Linear(input_dim, out_features), 'output_size': out_features}

    def activation_function(self, layer_num, input_size):
        activation_function_name = self.trial.suggest_categorical("activation_function_{}".format(layer_num),
                                                                  ["ReLU", "Sigmoid", "LeakyReLU", "tanh"])
        if activation_function_name == 'ReLU':
            layer = nn.ReLU()
        if activation_function_name == 'Sigmoid':
            layer = nn.Sigmoid()
        if activation_function_name == 'LeakyReLU':
            leaky_angle = self.trial.suggest_int("leaky_angle_l{}".format(layer_num), 1, 15)
            leaky_angle = 5 * leaky_angle
            layer = nn.LeakyReLU(leaky_angle)
        if activation_function_name == 'tanh':
            layer = nn.Tanh()
        if activation_function_name == 'Nothing':
            return

        return {'layer': layer, 'output_size': input_size}

    def dropout(self, layer_num, input_size):
        p = self.trial.suggest_int("dropout_{}".format(layer_num), 2, 5)
        p = 0.1 * p
        return {'layer': nn.Dropout(p), 'output_size': input_size}

    def layer_norm(self, layer_num, input_size):
        eps = self.trial.suggest_float("norm_eps_{}".format(layer_num), 1e-06, 1e-04, log=True)
        return {'layer': nn.LayerNorm(input_size, eps=eps), 'output_size': input_size}

    def skip_connection(self, x, layer):
        return torch.cat([layer(x), x], dim=-1)

    def forward(self, x):
        # print(f"Initial x shape: {list(x.shape)}")
        for i, (layer_dict, layer) in enumerate(zip(self.network, self.network_layers)):
            if layer_dict['layer_type'] == 'Activation':
                if layer_dict['skip_connection']:
                    x = self.skip_connection(x, layer)
                else:
                    x = layer(x)

            if layer_dict['layer_type'] == 'Dropout':
                if layer_dict['skip_connection']:
                    x = self.skip_connection(x, layer)
                else:
                    x = layer(x)

            if layer_dict['layer_type'] == 'BatchNorm':
                if layer_dict['skip_connection']:
                    x = self.skip_connection(x, layer)
                else:
                    x = layer(x)

            if layer_dict['layer_type'] == 'Linear':
                if layer_dict['skip_connection']:
                    x = self.skip_connection(x, layer)
                else:
                    x = layer(x)

            # print(f"Layer #{i} {layer_dict['layer_type']}, x shape: {list(x.shape)}")
        if self.det_or_prob == 'deterministic':
            return x
        else:
            mean, variance = torch.split(x, int(self.output_feature_size * 0.5), dim=1)
            variance = torch.nn.functional.softplus(variance) + self.trial.suggest_float('variance bias', 1e-7, 1e-5,
                                                                                         log=True)
            return mean, variance
