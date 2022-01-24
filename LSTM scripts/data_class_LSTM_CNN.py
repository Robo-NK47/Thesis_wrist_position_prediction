import torch
from tqdm.auto import tqdm
import numpy as np
from torch import nn
import random
import optuna


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

    def shape(self):
        return {'x': self.x.shape, 'y': self.y.shape}


class DataSet(torch.utils.data.Dataset):
    def __init__(self, general_set: dict(), imu_settings: dict(),
                 marker_settings: dict(), data_set: list(), transform=None):
        self.data_set = data_set
        self.excluded_samples = []
        self.sample_amount = len(self.data_set)
        self.preprocessing_settings = general_set
        self.imu_settings = imu_settings
        self.marker_settings = marker_settings
        self.transform = transform
        self.sample_shape = self.shape(self.data_set[0].x, self.data_set[0].y)
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
        return np.concatenate([sample, time_tags], axis=1)

    def enable_time_step(self):
        self.time_tag_x = True
        if self.time_tag_x:
            self.sample_shape['x'] = (self.sample_shape['x'][0], self.sample_shape['x'][1] + 1)

    def fetch_item(self, data, data_settings, preprocessing_settings):
        return_item = []
        data = data.transpose()
        counter = 0
        if preprocessing_settings['PCA']['to_preform'] and ('Gyroscope_wrist' in data_settings):
            return data.transpose()
        for sensor in data_settings:
            for axis in data_settings[sensor]:
                if data_settings[sensor][axis]:
                    return_item.append(data[counter])
                counter += 1

        return np.array(return_item).transpose()

    def shape(self, x, y):
        return {'x': self.fetch_item(x, self.imu_settings, self.preprocessing_settings).shape,
                'y': self.fetch_item(y, self.marker_settings, self.preprocessing_settings).shape}

    def allowed_samples(self, current_error, threshold, current_samples):
        all_cells = self.all_cells
        all_time_ranges = self.all_time_ranges
        addition = 2
        if current_error < threshold:
            if isinstance(current_samples[0], tuple):
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
            time_range_condition = sample.time_range in self.allowed_times
            cell_condition = sample.target_cell in self.allowed_cells
            if time_range_condition and cell_condition:
                samples_to_dataset.append(sample)
            else:
                samples_to_exclude.append(sample)

        for sample in self.excluded_samples:
            time_range_condition = sample.time_range in self.allowed_times
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


class LSTM_CNN(nn.Module):
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    def __init__(self, trial, num_of_input_features, num_of_output_features, input_time_steps, output_time_steps,
                 batch_size, device, cnn_it):
        super(LSTM_CNN, self).__init__()
        self.trial = trial
        self.input_feature_size = num_of_input_features
        self.output_feature_size = num_of_output_features
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.batch_size = batch_size
        self.device = device
        self.LSTM_layers = nn.ModuleList()
        self.crap_net = False

        self.num_of_LSTM_layers = self.trial.suggest_int("num_of_LSTM_layers", 1, 3)
        if cnn_it:
            self.num_of_CNN_layers = self.trial.suggest_int("num_of_CNN_layers", 1, 3)
            self.num_of_FC_layers = self.trial.suggest_int("num_of_FC_layers", 1, 3)
        else:
            self.num_of_CNN_layers = 0
            self.num_of_FC_layers = 0
        self.LSTM_output_size = 0
        self.CNN_output_shape = 0
        self.fc_input_shape = 0
        self.network_layers = self.make_a_network()
        self.cnn_it = cnn_it
        if cnn_it:
            print(f"The LSTM's output shape is: {self.LSTM_output_size}\n")
            good_cnn = False
            cnn_count = 0
            while not good_cnn:
                try:
                    self.CNN_layers = self.CNN_net()
                    good_cnn = True
                except RuntimeError:
                    good_cnn = False

                cnn_count += 1
                if cnn_count > 10:
                    self.crap_net = True
                    good_cnn = True
            if not self.crap_net:
                self.calc_fc_input_shape()
                print(f"\nThe CNN's output shape is: {self.CNN_output_shape}\n")
                self.FC_layers = self.FC_net(trial)

    def make_a_network(self):
        input_size = (self.batch_size, self.input_time_steps, self.input_feature_size)
        all_layers = []
        layer_types = ['LSTM', 'Activation', 'Dropout', 'BatchNorm', 'Skip']
        print("Generating the LSTM components.")
        for layer_num in tqdm(range(self.num_of_LSTM_layers)):
            for layer_type in layer_types:
                if layer_type == 'LSTM':
                    pre_lstm_size = input_size

                input_s = input_size
                layer = self.layer_chooser(layer_type, input_size, layer_num, pre_lstm_size)

                if layer['layer'] is not None:
                    input_size = layer['output_size']
                    layer = layer['layer']

                    layer_dict = {'layer_type': layer_type,
                                  'layer_number': len(all_layers),
                                  'input_size': input_s,
                                  'output_size': input_size}
                    all_layers.append(layer_dict)
                    if layer_type == 'LSTM':
                        if layer_num == self.num_of_LSTM_layers - 1:
                            layer_dict['last'] = True
                        else:
                            layer_dict['last'] = False

                    self.LSTM_layers.append(layer)

        self.LSTM_output_size = input_size
        return all_layers

    def layer_chooser(self, layer_type, input_size, layer_num, pre_lstm_size):
        if layer_type == 'LSTM':
            layer = self.LSTM_layer(layer_num, input_size)

        if layer_type == 'Activation':
            layer = self.activation_function(layer_num, input_size)

        if layer_type == 'Dropout':
            layer = self.dropout(layer_num, input_size)

        if layer_type == 'BatchNorm':
            layer = self.layer_norm(layer_num, input_size[1:])
            layer = {'layer': layer, 'output_size': input_size}

        if layer_type == 'Skip':
            if layer_num == self.num_of_LSTM_layers - 1:
                layer = {'layer': nn.Dropout(0.1001), 'output_size': input_size, 'skip': False}
            else:
                to_skip = 0 #self.trial.suggest_int("skip_connection_{}".format(layer_num), 0, 1)
                if to_skip:
                    input_size = (input_size[0], input_size[1], pre_lstm_size[-1] + input_size[2])
                    layer = {'layer': nn.Dropout(0.9001), 'output_size': input_size, 'skip': True}
                else:
                    layer = {'layer': nn.Dropout(0.1001), 'output_size': input_size, 'skip': False}

        return layer

    def layer_output_finder(self, layer, input_size):
        layer_output = layer(torch.zeros(input_size))
        if isinstance(layer_output, tuple):
            layer_output = layer_output[1][0]

        return tuple(layer_output.shape)

    def LSTM_layer(self, layer_num, input_dim):
        hidden_dim = self.trial.suggest_int(f"lstm_hidden_layer_dim{layer_num}", 6, 9)
        hidden_dim = 2 ** hidden_dim
        num_layers = self.trial.suggest_int(f"lstm_num_layers{layer_num}", 1, 5)
        # self.num_layers = num_layers
        # self.hidden_dim = hidden_dim
        lstm_dropout = 0
        if num_layers > 1:
            lstm_dropout = self.trial.suggest_int(f"lstm_dropout{layer_num}", 2, 8)
            lstm_dropout = 0.1 * lstm_dropout
        LSTM_layer = nn.LSTM(input_size=input_dim[-1],
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=lstm_dropout,
                             bidirectional=True)
        LSTM_layer = LSTM_layer.to(self.device)
        output_size = LSTM_layer(torch.zeros(input_dim).to(self.device))
        #if layer_num == self.num_of_LSTM_layers - 1:
        output_size = output_size[1][0].shape
        output_size = (output_size[1], output_size[0], output_size[2])
        # else:
        #     output_size = output_size[0].shape
        to_return = {'layer': LSTM_layer, 'output_size': output_size, 'num_layers': num_layers}
        # print(f'LSTM #{layer_num} output size: {output_size}')
        return to_return

    def FC_layer(self, layer_num, input_dim):
        out_features = self.trial.suggest_int("n_FC_units_{}".format(layer_num), 6, 10)
        out_features = 2 ** out_features
        return {'layer': nn.Linear(input_dim, out_features), 'output_size': out_features}

    def activation_function(self, layer_num, input_size):
        activation_function_name = self.trial.suggest_categorical("activation_function_{}".format(layer_num),
                                                                  ["Sigmoid", "ReLU", "Nothing"])
        if activation_function_name == 'ReLU':
            layer = nn.ReLU()
        if activation_function_name == 'Sigmoid':
            layer = nn.Sigmoid()
        if activation_function_name == 'LeakyReLU':
            leaky_angle = self.trial.suggest_float("leaky_angle_l{}".format(layer_num), 0, 90)
            layer = nn.LeakyReLU(leaky_angle)
        if activation_function_name == 'Nothing':
            layer = None

        return {'layer': layer, 'output_size': input_size}

    def dropout(self, layer_num, input_size):
        p = self.trial.suggest_int("dropout_{}".format(layer_num), 2, 5)
        p = 0.1 * p
        return {'layer': nn.Dropout(p), 'output_size': input_size}

    def skip_connection(self, x, layer):
        return torch.cat([layer(x), x], dim=1)

    def init_weigths(self):
        for layer in self.LSTM_layers:
            if isinstance(layer, nn.LSTM):
                for w in layer.all_weights:
                    for k in w:
                        if k.dim() >= 2:
                            torch.nn.init.xavier_normal_(k, 1)

    def CNN_net(self):
        layers = []
        input_dim = torch.unsqueeze(torch.zeros(self.LSTM_output_size), dim=1).shape
        i = 0
        counter = 0
        # print(f'LSTM output size: {self.LSTM_output_size}')
        # ("Generating the convolutional components.")
        while i < self.num_of_CNN_layers:
            print(f'{counter + 1} attempt(s) to generate the convolutional components.', end='')
            # print(counter)
            current_layers, input_dim = self.CNN_block_maker(i, input_dim)

            if i == 0:
                _dim = torch.unsqueeze(torch.zeros(self.LSTM_output_size), dim=1).shape
                check = self.is_dim_ok(current_layers, _dim)
            else:
                check = self.is_dim_ok(current_layers, input_dim)

            if check['ok']:
                input_dim = check['dim']
                i += 1
                for layer in current_layers:
                    layers.append(layer)
            counter += 1
            if counter > 100:
                layers = []
                input_dim = torch.unsqueeze(torch.zeros(self.LSTM_output_size), dim=1).shape
                i = 0
                counter = 0

        # layers.append(nn.Linear(input_dim, self.LSTM_output_size))
        return nn.Sequential(*layers)

    def CNN_block_maker(self, i, input_dim):
        current_layers = []
        # if i == 0 and input_dim[-1] < 100:
        #     transpose_cnn = self.transpose_CNN_layer(i, input_dim, self.trial)
        #     current_layers.append(transpose_cnn['layer'])
        #     input_dim = transpose_cnn['output_size']

        cnn = {'layer': None}
        while cnn['layer'] is None:
            cnn = self.CNN_layer(i, input_dim, self.trial)
        current_layers.append(cnn['layer'])
        output_dim = cnn['output_size']

        pooling = {'layer': None}
        while pooling['layer'] is None:
            pooling = self.max_pooling(i, output_dim, self.trial)
        current_layers.append(pooling['layer'])

        activation = self.activation_function(i, self.trial)
        if activation['layer'] is not None:
            current_layers.append(activation['layer'])

        dropout = self.dropout(i, self.trial)
        current_layers.append(dropout['layer'])

        return current_layers, input_dim

    def calc_fc_input_shape(self):
        self.CNN_output_shape = self.calc_layer_output(self.CNN_layers,
                                                       torch.unsqueeze(torch.zeros(self.LSTM_output_size),
                                                                       dim=1).shape)

        lstm_faulty_output = torch.zeros(self.LSTM_output_size).to(self.device)
        cnn_faulty_output = torch.zeros(self.CNN_output_shape).to(self.device)

        x = self.connection_layer(lstm_faulty_output, cnn_faulty_output)  # With LSTM data

        self.fc_input_shape = torch.cat([x, cnn_faulty_output], dim=1).shape[1:]

    def FC_net(self, trial):
        layers = [nn.Flatten(start_dim=1)]

        out_size = self.fc_input_shape

        temp = 1
        for x in out_size:
            temp *= x
        out_size = temp  # With LSTM data
        print("Generating the fully connected components:")
        for i in tqdm(range(self.num_of_CNN_layers, self.num_of_CNN_layers + self.num_of_FC_layers)):
            if i < self.num_of_CNN_layers + self.num_of_FC_layers - 1:
                fc_layer = self.FC_layer(i + 1, out_size)
                out_size = fc_layer['output_size']
                layers.append(fc_layer['layer'])

                activation_function = self.activation_function(i + 1, trial)
                if activation_function['layer'] is not None:
                    layers.append(activation_function['layer'])

                layers.append(self.dropout(i + 1, trial)['layer'])
                layer_norm = self.layer_norm(i + 1, out_size)
                if layer_norm is not None:
                    layers.append(layer_norm)

            else:
                layers.append(nn.Linear(out_size, self.output_feature_size))

        return nn.Sequential(*layers)

    def Flatten(self, input_dim):
        layer = nn.Flatten()
        output_dim = layer(torch.zeros(input_dim)).shape
        return {'layer': layer,
                'output_size': output_dim}

    def CNN_layer(self, layer_num, input_dim, trial):
        dim_ok = False
        counter = 0
        kernel_size_x = trial.suggest_int("conv_kernel_size_x{}".format(layer_num), 1, 5)
        kernel_size_y = trial.suggest_int("conv_kernel_size_y{}".format(layer_num), 1, 5)
        stride = trial.suggest_int("conv_stride_l{}".format(layer_num), 1, 3)
        out_channels = trial.suggest_int("out_channels_{}".format(layer_num), 3, 8)
        while not dim_ok:

            while not dim_ok:
                kernel_size_x = random.randint(1, 5)
                kernel_size_y = random.randint(1, 5)
                stride = random.randint(1, 3)
                out_channels = random.randint(3, 10)

                optuna.trial.FixedTrial({"conv_kernel_size_x{}".format(layer_num): kernel_size_x,
                                         "conv_kernel_size_y{}".format(layer_num): kernel_size_y,
                                         "conv_stride_l{}".format(layer_num): stride,
                                         "out_channels_{}".format(layer_num): out_channels})
                if not (kernel_size_x == 1 and kernel_size_y == 1):
                    break
            # print(f'Kernel size - {(kernel_size_x, kernel_size_y)}')
            if layer_num == 0:
                out_channels = 1

            layer = nn.Conv2d(in_channels=input_dim[1], out_channels=out_channels,
                              kernel_size=(kernel_size_x, kernel_size_y), stride=stride)
            try:
                output_dim = layer(torch.zeros(input_dim)).shape
                dim_ok = True
                # print(f'Internal CNN layer print, output dim - {output_dim}')
                if output_dim[-1] < 2:
                    dim_ok = False
                    counter += 1
            except RuntimeError:
                dim_ok = False
                counter += 1
            if counter > 100:
                return {'layer': None, 'output_size': None}
        return {'layer': layer, 'output_size': output_dim}

    def transpose_CNN_layer(self, layer_num, input_dim, trial):
        stride = trial.suggest_int("trans_conv_stride{}".format(layer_num), 5, 12)
        padding = trial.suggest_int("trans_conv_padding{}".format(layer_num), 1, 3)
        kernel_size = trial.suggest_int("trans_conv_kernel_size{}".format(layer_num), 3, 6)
        in_channels = input_dim[1]
        out_channels = trial.suggest_int("trans_conv_kernel_out_channel{}".format(layer_num), input_dim[0],
                                         input_dim[0] * 3)

        layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        fake_input = torch.zeros(input_dim)
        output_dim = layer(fake_input).shape

        return {'layer': layer, 'output_size': output_dim}

    def max_pooling(self, layer_num, input_dim, trial):
        dim_ok = False
        counter = 0
        stride = trial.suggest_int("max_pool_stride_l{}".format(layer_num), 2, 3)
        max_pool_size_x = trial.suggest_int("max_pool_size_x{}".format(layer_num), 1, 4)
        max_pool_size_y = trial.suggest_int("max_pool_size_y{}".format(layer_num), 1, 4)
        while not dim_ok:
            stride = random.randint(2, 3)
            max_pool_size_x = random.randint(1, 4)
            max_pool_size_y = random.randint(1, 4)

            optuna.trial.FixedTrial({"max_pool_size_x{}".format(layer_num): max_pool_size_x,
                                     "max_pool_size_y{}".format(layer_num): max_pool_size_y,
                                     "max_pool_stride_l{}".format(layer_num): stride})
            layer = nn.MaxPool2d((max_pool_size_x, max_pool_size_y), stride=stride)
            try:
                output_dim = layer(torch.zeros(input_dim)).shape
                dim_ok = True
            except RuntimeError:
                dim_ok = False
                counter += 1
            if counter > 100:
                return None

        return {'layer': layer,
                'output_size': output_dim}

    def calc_layer_output(self, layer, last_input_dim):
        return layer(torch.zeros(last_input_dim)).shape

    def is_dim_ok(self, layers, input_dim):
        layers = nn.Sequential(*layers)
        fake_input = torch.zeros(input_dim)

        fake_output = layers(fake_input)
        if fake_output.shape[-1] >= 4:
            dim_ok = True
        else:
            dim_ok = False

        return {'ok': dim_ok, 'dim': fake_output.shape}

    def layer_norm(self, layer_num, input_size):
        eps = self.trial.suggest_float("norm_eps_{}".format(layer_num), 1e-06, 1e-04, log=True)
        return nn.LayerNorm(input_size, eps=eps)

    def connection_layer(self, lstm_x, cnn_x):
        in_features = lstm_x.shape[-1] * lstm_x.shape[-2]
        out_features = cnn_x.shape[-1] * cnn_x.shape[-2] * cnn_x.shape[-3]
        flatten = nn.Flatten()
        linear = nn.Linear(in_features, out_features, device=self.device)

        lstm_x = flatten(lstm_x)
        lstm_x = linear(lstm_x)

        return lstm_x.view(cnn_x.shape)

    def forward(self, x):
        # print(f"Initial x shape: {list(x.shape)}")
        for i, (layer_dict, layer) in enumerate(zip(self.network_layers, self.LSTM_layers)):
            # print(f"Layer #{i}, layer type: {layer_dict['layer_type']}, input shape: {x.shape}")
            if layer_dict['layer_type'] == 'LSTM':
                x_pre_LSTM = x
                if layer_dict['last']:
                    x_n, (hidden_state, initial_cell_state) = layer(x)
                    x = hidden_state
                    x = x.permute((1, 0, 2))
                else:
                    # x_n, (hidden_state, initial_cell_state) = layer(x)
                    # x = x_n
                    x_n, (hidden_state, initial_cell_state) = layer(x)
                    x = hidden_state
                    x = x.permute((1, 0, 2))

            if layer_dict['layer_type'] == 'Activation':
                x = layer(x)

            if layer_dict['layer_type'] == 'Dropout':
                x = layer(x)

            if layer_dict['layer_type'] == 'BatchNorm':
                x = layer(x)

            if layer_dict['layer_type'] == 'Skip':
                if layer.p == 0.9001 and x.shape[:2] == x_pre_LSTM.shape[:2]:
                    x = torch.cat([x, x_pre_LSTM], dim=-1)
            # print(f"Output shape: {list(x.shape)}\n")
        if self.cnn_it:
            x_cnn = self.CNN_layers(torch.unsqueeze(x, dim=1))

            x = self.connection_layer(x, x_cnn)  # With LSTM data

            x = torch.cat([x, x_cnn], dim=1)

            x = self.FC_layers(x)  # x # With LSTM data
        else:
            flat = nn.Flatten(start_dim=1)
            x = flat(x)
            layer = nn.Linear(x.shape[-1], self.output_feature_size).to(self.device)
            x = layer(x)
        return torch.unsqueeze(x, 1)
