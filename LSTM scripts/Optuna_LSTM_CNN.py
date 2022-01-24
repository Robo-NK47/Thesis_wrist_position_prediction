import os
import time
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.utils.data
import pickle
from tqdm.auto import tqdm
from collections import defaultdict
import shutil
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
import openpyxl
import joblib
from data_class_LSTM_CNN import LSTM_CNN
from data_class_LSTM import DataSet, DataSample, ToTensor, LSTM
from distutils.dir_util import copy_tree
from shutil import copy
from numba import cuda as CUDA

# import see_data_LSTM


def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


def load_file(path):
    with (open(path, "rb")) as openfile:
        try:
            return pickle.load(openfile)
        except EOFError:
            return 'File error.'


def save(pat, data):
    with open(pat, 'wb') as f:
        pickle.dump(data, f)


def loss_graph(delta_list, graph_path, train_lss, test_lss, trial):
    keys = delta_list[0].keys()
    new_dict = {}.fromkeys(keys)
    trial_num = trial.number

    for key in keys:
        sample = []
        for i in range(len(delta_list)):
            sample.append(delta_list[i][key])
        new_dict[key] = sample

    fig, axs = plt.subplots(2, figsize=(20, 20))
    fig.suptitle(f'Loss & variance graphs')

    xs = range(1, len(new_dict[key]) + 1)

    for key in new_dict:
        axs[0].set_title('Variance - The total variance in the all of the output data.')
        axs[0].plot(xs, new_dict[key], label=f'Sample {key}')
        axs[0].set_ylabel('variance')
        axs[0].legend(loc="upper left")
        for x, y in zip(xs, new_dict[key]):
            if x % int(len(xs) * 0.25) == 0:
                axs[0].annotate(f'{y:.5E}', xy=(x, y))

    axs[1].set_title('Loss')
    axs[1].plot(xs, train_lss, label='Train loss')
    for x, y in zip(xs, train_lss):
        if x % int(len(xs) * 0.25) == 0:
            axs[1].annotate(f'{y:.5E}', xy=(x, y))
    axs[1].plot(xs, test_lss, label='Test loss')
    for x, y in zip(xs, test_lss):
        if x % int(len(xs) * 0.25) == 0:
            axs[1].annotate(f'{y:.5E}', xy=(x, y))

    axs[1].set_ylabel('Loss [mm]')
    axs[1].legend(loc="upper left")

    path = os.path.join(graph_path, f'Loss & variance graph - trial #{trial_num}.png')
    plt.tight_layout(pad=3.0)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')


def get_dataset(batch_size):
    main_path = dataset_path
    ####################### TRAIN DATASET #######################
    train_data_path = os.path.join(main_path, 'train dataset - LSTM.pkl')
    train_data = load_file(train_data_path)
    train_data.transform = ToTensor()
    ############################################################

    ####################### TEST DATASET #######################
    test_data_path = os.path.join(main_path, 'test dataset - LSTM.pkl')
    test_data = load_file(test_data_path)
    test_data.transform = ToTensor()
    ############################################################
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    del train_data, test_data

    return train_loader, test_loader


def calc_variance(net_outputs):
    net_outputs = net_outputs[0]
    output_var = net_outputs.cpu().detach().numpy()

    return {'variance': np.sum(np.var(output_var, axis=0))}


def calc_delta(net_outputs, labels):
    shaper = labels.shape[0]
    delta = torch.sum((labels - net_outputs) ** 2) ** 0.5
    return delta / shaper


def graph_predictions(test_data, model, save_path, epoc, settings):
    input_length = len(test_data.dataset[0][0])
    label_length = len(test_data.dataset[0][1])
    num_of_input_features = test_data.dataset[0][0].shape[1]
    num_of_output_features = test_data.dataset[0][1].shape[1]
    epoc += 1

    settings_dict = settings['test_y']
    feature_names = []
    for i, feature in enumerate(settings_dict):
        for j, axis in enumerate(settings_dict[feature]):
            if settings_dict[feature][axis]:
                feature_names.append(f'{feature}-{axis}')

    x = range(label_length)
    for r, sample in enumerate(test_data.dataset):
        precs = [0, 0.25, 0.5, 0.75, 0.99]
        condition_list = [False for prec in precs if int(prec * len(test_data.dataset)) == r]
        if not all(condition_list):
            prediction_label = sample[1].reshape(label_length, num_of_output_features)
            net_output, h = model(sample[0].reshape(1, input_length, num_of_input_features).float().to(device))
            net_output = net_output.reshape(label_length, num_of_output_features)

            ground_truths = prediction_label.transpose(0, 1)
            predictions = net_output.transpose(0, 1)
            axis_num = 3

            fig, axs = plt.subplots(nrows=int(len(feature_names) / axis_num), ncols=axis_num,
                                    figsize=(20, 10 * int(len(feature_names) / axis_num)))
            fig.suptitle(f'Sample #{r} - epoch #{epoc}')
            for feature_name, ground_truth, prediction, ax in zip(feature_names, ground_truths, predictions, axs.flat):
                ax.set_title(feature_name, loc='left')
                ax.plot(x, prediction.cpu().detach().numpy(), label='prediction')
                ax.plot(x, ground_truth.cpu().detach().numpy(), label='Ground truth')
                ax.legend(loc="upper left")
                ax.set_ylim([0, 1])

            path = os.path.join(save_path, f'Sample #{r} - epoch #{epoc}.png')
            fig.tight_layout(pad=3.0)
            fig.savefig(path, dpi=100, bbox_inches='tight')
            plt.close('all')


def make_text_file(trial, mdl_path, trn_losses, tst_losses, var_list, mdl, gen_settings, dataset_settings,
                   train_shp, test_shp):
    trial_parameters = trial.params
    text_path = os.path.join(mdl_path, 'trial parameters.txt')
    param_list = ['Trial parameters:\n']
    for parameter in trial_parameters:
        param_list.append(f"{parameter}: {trial_parameters[parameter]}\n")

    param_list.append("\nDataset settings:\n")
    for ky in gen_settings:
        param_list.append(f"{ky}: {gen_settings[ky]}\n")
    param_list.append(f"\nTrain data shape: {train_shp}")
    param_list.append(f"\nTest data shape: {test_shp}\n")

    param_list.append("\nAdditional dataset settings:\n")
    IMU_settings = dataset_settings['train_x']
    marker_settings = dataset_settings['train_y']
    param_list.append('\nIMU:\n')
    for ky in IMU_settings:
        param_list.append(f"{ky}: {IMU_settings[ky]}\n")
    param_list.append('\nMarker:\n')
    for ky in marker_settings:
        param_list.append(f"{ky}: {marker_settings[ky]}\n")

    param_list.append("\n")
    param_list.append("\n\nTrial final  results:\n")
    if len(trn_losses) > 0:
        param_list.append(f"Train loss: {trn_losses[-1]}\n")
    else:
        param_list.append(f"Train loss: Incomplete\n")

    if len(tst_losses) > 0:
        param_list.append(f"Test loss: {tst_losses[-1]}\n")
    else:
        param_list.append(f"Test loss: Incomplete\n")

    if len(tst_losses) > 0:
        param_list.append(f"Network data variance: {var_list[-1]}\n")
    else:
        param_list.append(f"Network data variance: Incomplete\n")

    param_list.append("\n\nAll trial results:\n")
    param_list.append(f"\n\nTrain losses: {trn_losses}\n")
    param_list.append(f"\n\nTest losses: {tst_losses}\n")
    param_list.append(f"\n\nNetwork data variance: {var_list}\n")

    with open(text_path, "w") as text_file:
        text_file.writelines(param_list)


def delta_heatmap(errors, graph_path, epoch, trial):
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    rows = [0, 1, 2, 3, 4, 5]

    trial_num = trial.number
    time_frames_to_plot = list(errors.index)

    time_frames_to_plot = [time_frames_to_plot[0], time_frames_to_plot[int(0.5 * len(time_frames_to_plot))],
                           time_frames_to_plot[-1]]

    all_data_frames = {}.fromkeys(time_frames_to_plot)
    for time_frame_to_plot in all_data_frames:
        data_frame = pd.DataFrame(data=np.zeros((len(rows), len(columns))), index=rows, columns=columns, dtype=float)
        for cell_letter in columns:
            for cell_number in rows:
                if f'({cell_letter}, {cell_number})' in errors:
                    data_frame[cell_letter][cell_number] = errors[f'({cell_letter}, {cell_number})'][time_frame_to_plot]
        all_data_frames[time_frame_to_plot] = data_frame

    min_values = []
    max_values = []
    mean_values = []
    for df in all_data_frames:
        dataframe = all_data_frames[df]

        min_value = str(dataframe.min().min())[0:6]
        min_values.append(dataframe.min().min())
        max_value = str(dataframe.max().max())[0:6]
        max_values.append(dataframe.max().max())
        mean_value = str(dataframe.mean().mean())[0:6]
        mean_values.append(dataframe.mean().mean())

        fig, ax = plt.subplots(2, figsize=(12, 20))
        sns.heatmap(dataframe, robust=True, annot=True, vmin=0, vmax=100, ax=ax[0])
        sns.heatmap(dataframe, robust=True, annot=True, ax=ax[1])
        fig.suptitle(f'Error heat map, epoch {epoch}, time step {df}\n'
                     f'Mean value: {mean_value}, min value: {min_value}, max value: {max_value}')
        path = os.path.join(graph_path, f'Trial #{trial_num} - time step #{df} - epoch #{epoch}.png')
        plt.tight_layout(pad=3.0)
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close('all')

    plt.close('all')
    return {'mean': np.array(mean_values),
            'min': np.array(min_values),
            'max': np.array(max_values),
            'epoch': epoch,
            'time_steps': time_frames_to_plot}


def heat_map_summary(data, path):
    means = []
    mins = []
    maxs = []
    epochs = []
    time_steps = data[0]['time_steps']

    for epoch in data:
        means.append(epoch['mean'])
        mins.append(epoch['min'])
        maxs.append(epoch['max'])
        epochs.append(epoch['epoch'])

    means = pd.DataFrame(data=np.array(means), index=epochs, columns=time_steps, dtype=float)
    mins = pd.DataFrame(data=np.array(mins), index=epochs, columns=time_steps, dtype=float)
    maxs = pd.DataFrame(data=np.array(maxs), index=epochs, columns=time_steps, dtype=float)

    max_val = maxs.max().max()

    param_names = ['max', 'mean', 'min']

    fig, axes = plt.subplots(3, figsize=(15, 20))
    fig.suptitle(f'Error parameters.')
    mins.plot(ax=axes[2])
    means.plot(ax=axes[1])
    maxs.plot(ax=axes[0])

    for i, parameter in enumerate(param_names):
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Error')
        # axes[i].set_ylim([0, max_val])
        axes[i].set_title(parameter)
        axes[i].get_legend().remove()
        # axes[i].legend(loc="upper left", bbox_to_anchor=[0, 1], shadow=True, title="Time step", fancybox=True)

    path = os.path.join(path, f'Error parameters.png')
    plt.tight_layout(pad=3.0)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')


def calc_graph_limits(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    means = {'x': np.mean(x),
             'y': np.mean(y),
             'z': np.mean(z)}

    gaps = [np.abs(np.max(x) - np.min(x)),
            np.abs(np.max(y) - np.min(y)),
            np.abs(np.max(z) - np.min(z))]

    max_gap = max(gaps) * 0.5
    more_gap = 100
    limits = {'x': [(means['x'] - max_gap - more_gap), means['x'] + max_gap + more_gap],
              'y': [(means['y'] - max_gap - more_gap), means['y'] + max_gap + more_gap],
              'z': [(means['z'] - max_gap - more_gap), means['z'] + max_gap + more_gap]}

    return limits


def trajectory_plot(data, g_path, trial, epc):
    trial_num = trial.number
    selected_cells = ['A, 0', 'B, 1', 'C, 2', 'D, 3', 'E, 4', 'F, 5']
    for selected_cell in selected_cells:
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        real_x = []
        real_y = []
        real_z = []

        predicted_x = []
        predicted_y = []
        predicted_z = []
        for cell in sorted(data.keys()):
            if cell[1:5] == selected_cell:
                for time_frame in data[cell]:
                    if len(data[cell][time_frame]['ground_truth']) > 0:
                        current_ground_truth = data[cell][time_frame]['ground_truth'][0]
                        current_prediction = data[cell][time_frame]['prediction'][0]

                        for ground, pred in zip(current_ground_truth.tolist(),
                                                current_prediction.tolist()):
                            real_x.append(ground[0])
                            real_y.append(ground[1])
                            real_z.append(ground[2])

                            predicted_x.append(pred[0])
                            predicted_y.append(pred[1])
                            predicted_z.append(pred[2])

        predicted_graph_limits = calc_graph_limits(predicted_x, predicted_y, predicted_z)
        real_graph_limits = calc_graph_limits(real_x, real_y, real_z)

        x = [min(predicted_graph_limits['x'][0], real_graph_limits['x'][0]),
             max(predicted_graph_limits['x'][1], real_graph_limits['x'][1])]

        y = [min(predicted_graph_limits['y'][0], real_graph_limits['y'][0]),
             max(predicted_graph_limits['y'][1], real_graph_limits['y'][1])]

        z = [min(predicted_graph_limits['z'][0], real_graph_limits['z'][0]),
             max(predicted_graph_limits['z'][1], real_graph_limits['z'][1])]
        graph_limits = {'x': x, 'y': y, 'z': z}

        first_pred = [predicted_x[0], predicted_y[0], predicted_z[0]]
        target = [real_x[-1], real_y[-1], real_z[-1]]

        for time_step, (x, y, z) in enumerate(zip(predicted_x, predicted_y, predicted_z)):
            prediction = [x, y, z]
        c = []
        for kak in range(len(predicted_x)):
            n = 1 - 0.5 * (kak / len(predicted_x))
            c.append([n, 1 - n, n])

        ax.scatter3D(real_x, real_y, real_z, c=c, marker='^', label=f'Ground truths')

        c = []
        for kak in range(len(predicted_x)):
            n = 0.5 * (kak / len(predicted_x))
            c.append([n, 1 - n, n])

        ax.scatter3D(predicted_x, predicted_y, predicted_z, c=c, marker='X', label=f'Predictions')

        ax.set_title(f'End point, reaching cell ({selected_cell}, epoch - {epc})\n')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')

        # ax.set_xlim(graph_limits['x'])
        # ax.set_ylim(graph_limits['y'])
        # ax.set_zlim(graph_limits['z'])

        ax.legend()

        path = os.path.join(g_path, f'Trial #{trial_num} - cell #{selected_cell} - '
                                    f'epoch #{epc} (end point graph).png')

        plt.savefig(path, dpi=100, format='png')
        plt.close('all')


def error_thru_time_plot(all_errors_by_cell, g_path, trial, epc):
    trial_num = trial.number
    selected_cells = ['A, 0', 'B, 1', 'C, 2', 'D, 3', 'E, 4', 'F, 5']

    list_of_timestep_means = list(all_errors_by_cell.mean(1))
    list_of_timestep_STDs = list(all_errors_by_cell.std(1))

    total_mean = all_errors_by_cell.mean().mean()
    time_steps = []
    for time_step in list(all_errors_by_cell.index):
        time_steps.append(str(time_step))

    fig, axs = plt.subplots(2, figsize=(20, 20))
    for selected_cell in selected_cells:
        cell = f'({selected_cell})'
        axs[0].plot(time_steps, list(all_errors_by_cell[cell]), label=cell)

    axs[0].legend(loc="upper left", title="Selected matrix cell")
    axs[0].set_title('Error as a function of time step for selected matrix cells')
    axs[0].set_ylabel('Error [mm]')
    axs[0].set_xlabel('Time step')

    axs[1].errorbar(time_steps, list_of_timestep_means, yerr=list_of_timestep_STDs, color='black',
                    ecolor='red')
    axs[1].set_title(f'Mean error as a function of time step, overall mean - {total_mean:.6f} [mm]')
    axs[1].set_ylabel('Error [mm]')
    axs[1].set_xlabel('Time step')

    plt.setp(axs[0].get_xticklabels(), rotation='vertical')
    plt.setp(axs[1].get_xticklabels(), rotation='vertical')

    fig.suptitle(f'Error as a function of time step - epoch {epc}')
    path = os.path.join(g_path, f'Trial #{trial_num} - Errors along time steps - epoch {epc}.png')

    plt.savefig(path, dpi=100, format='png')
    plt.close('all')


def calc_error(ground_truth, prediction):
    MSE = nn.MSELoss(reduction='sum')
    return torch.sqrt(MSE(ground_truth, prediction))


def return_unique_cells_and_times(cells, time_ranges):
    cells = sorted(list(({}.fromkeys(cells)).keys()))
    time_ranges = sorted(list(({}.fromkeys(time_ranges)).keys()))
    return cells, time_ranges


def make_neat_epoch_data(data):
    all_cells = []
    all_times = []
    for sample in data:
        all_cells.append(sample['cell'])
        all_times.append(sample['time_range'])

    all_cells, all_times = return_unique_cells_and_times(all_cells, all_times)
    return_item = {}.fromkeys(all_cells)
    for cell in tqdm(all_cells):
        cell_dict = {}.fromkeys(all_times)
        for time_range in all_times:
            a = defaultdict(list)
            for sample in data:
                target_cell = sample['cell']
                time_ = sample['time_range']
                if target_cell == cell and time_range == time_:
                    a['prediction'].append(sample['prediction'])
                    a['ground_truth'].append(sample['ground_truth'])
                    a['errors'].append(sample['error'])
            a['error'] = sum(a['errors']) / len(a['errors'])
            cell_dict[time_range] = a
        return_item[cell] = cell_dict

    return return_item


def make_an_error_data_frame(data):
    datas = {}.fromkeys(data.keys())

    for cell in data:
        cell_data = {}.fromkeys(data[cell].keys())
        for time_range in data[cell]:
            cell_data[time_range] = data[cell][time_range]['error']
        datas[cell] = cell_data

    return pd.DataFrame.from_dict(datas)


def excel(path, action, data):
    if action == 'create':
        wb = openpyxl.Workbook()
        ws = wb.active
        headers = ['#', 'LSTM layers', 'CNN layers', 'FC layers','LR', 'Weight decay', 'Best train loss',
                   'Best train loss epoch', 'Last train loss', 'Best test loss', 'Best test loss epoch',
                   'Last test loss', 'Last output data variance', 'Optuna minimization factor',
                   'How many time the network has minimized its minimization factor', 'Best error thru time']

        ws.append(headers)

        path = os.path.join(path, 'Results xl.xlsx')
        wb.save(path)

    if action == 'update':
        path = os.path.join(path, 'Results xl.xlsx')
        wb = openpyxl.load_workbook(filename=path)
        ws = wb.active

        dat = [data['trial'].number,
               data['model'].num_of_LSTM_layers,
               data['model'].num_of_CNN_layers,
               data['model'].num_of_FC_layers,
               data['trial'].params['lr'],
               data['trial'].params['weight_decay'],
               min(data['train_losses']),
               data['train_losses'].index(min(data['train_losses'])),
               data['train_losses'][-1],
               min(data['test_losses']),
               data['test_losses'].index(min(data['test_losses'])),
               data['test_losses'][-1],
               data['variance_list'][-1]['variance'],
               data['to_minimize'],
               data['minimization_counter'],
               data['error_thru_time']]

        ws.append(dat)
        wb.save(path)


def clear_memory_all_gpus():
    """Clear memory of all GPUs."""
    try:
        for gpu in CUDA.gpus:
            with gpu:
                CUDA.current_context().deallocations.clear()
    except CUDA.CudaSupportError:
        print("No CUDA available")


def objective(trial):
    global operating_system
    global trial_start
    global study_path
    # Get the dataset.
    batch_size = 8  # trial.suggest_int('batch_size', 8, 8)
    batch_size = 2 ** batch_size
    train_dataset, test_dataset = get_dataset(batch_size)

    allowed_cells = train_dataset.dataset.all_cells
    train_dataset.dataset.allowed_cells = allowed_cells
    test_dataset.dataset.allowed_cells = allowed_cells

    allowed_times = train_dataset.dataset.all_time_ranges[0:10]
    train_dataset.dataset.allowed_times = allowed_times
    test_dataset.dataset.allowed_times = allowed_times

    train_dataset.dataset.exclude_samples()
    test_dataset.dataset.exclude_samples()
    train_dataset.sampler.data_source = train_dataset.dataset.data_set
    test_dataset.sampler.data_source = test_dataset.dataset.data_set

    dataset_settings = {'train_x': train_dataset.dataset.imu_settings,
                        'train_y': train_dataset.dataset.marker_settings,
                        'test_x': test_dataset.dataset.imu_settings,
                        'test_y': test_dataset.dataset.marker_settings}

    general_settings = train_dataset.dataset.preprocessing_settings

    best_trial_score = 10000
    smallest_error_thru_time = 10000

    minimization_counter = 1
    if len(trial.study.best_trials) > 0:
        if best_trial_score > trial.study.best_value:
            t_path = os.path.join(study_path, f'study score - best model is {trial.study.best_trials[-1].number}.txt')
            t_dict = [f'Best trial - {trial.study.best_trials[-1].number}',
                      f'\n\nBest score - {trial.study.best_value}',
                      f'\n\nBest trial parameters - {trial.study.best_params} '
                      f'with a score of - {trial.study.best_value:.2f}']
            with open(t_path, "w") as text_file:
                text_file.writelines(t_dict)

    train_losses = []
    test_losses = []
    variance_list = []
    heatmap_values = []

    # Generate the model.
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    crap_model = True
    with_cnn = 1  # trial.suggest_int('with cnn part', 0, 1)
    while crap_model:
        model = LSTM_CNN(trial=trial,
                         num_of_input_features=train_dataset.dataset.data_set[0].x.shape[-1],
                         num_of_output_features=train_dataset.dataset.sample_shape['y'][-1],
                         input_time_steps=train_dataset.dataset.data_set[0].x.shape[0],
                         output_time_steps=train_dataset.dataset.sample_shape['y'][0],
                         batch_size=train_dataset.batch_size,
                         device=device,
                         cnn_it=with_cnn).to(device)

        crap_model = model.crap_net

    model.init_weigths()
    print(f'\n\n{model}\n')
    print(f'Amount of train samples: {len(train_dataset.dataset)}')
    print(f'Amount of test samples: {len(test_dataset.dataset)}')
    print(f'\nInput_sample_size: {train_dataset.dataset.sample_shape["x"]}\n'
          f'Output_sample_size: {train_dataset.dataset.sample_shape["y"]} \n\n')

    loss_dict = {"MSE": nn.MSELoss(reduction='none'), "L1": nn.L1Loss()}
    criterion = loss_dict["MSE"]

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=0, factor=0.8,
                                                           min_lr=1e-7, verbose=True)

    train_shape = {'sample_amount': len(train_dataset.dataset),
                   'shape': train_dataset.dataset.sample_shape["x"]}
    test_shape = {'sample_amount': len(test_dataset.dataset),
                  'shape': train_dataset.dataset.sample_shape["y"]}

    # Training of the model.
    epochs = 10
    epoch = 0
    since_increased = 0
    epoch_counter = 0
    # Training of the model.
    while epoch < epochs:
        print(f'Epoch #{epoch_counter} (last improved {epoch} epoch[s] ago)\n'
              f'({dataset_type}) - Time range: {allowed_times[0]} to {allowed_times[-1]}')
        model.train()
        train_loss = 0
        if epoch_counter == 0:
            model_path = os.path.join(study_path, f'Model #{trial.number}')
            trajectory_graph_path = os.path.join(model_path, 'trajectory graphs')
            heatmap_graph_path = os.path.join(model_path, 'heatmap graphs')
            error_thru_time_plot_path = os.path.join(model_path, 'error thru time')
            os.mkdir(model_path)
            os.mkdir(trajectory_graph_path)
            os.mkdir(heatmap_graph_path)
            os.mkdir(error_thru_time_plot_path)

        print('     Training:')
        for batch_idx, (data, target, sample_indexes) in enumerate(tqdm(train_dataset)):
            data, target = data.to(device).float(), target.to(device).float()

            optimizer.zero_grad()
            output = model(data)

            loss = torch.mean(torch.sqrt(torch.sum(criterion(output, target), dim=1)), dtype=torch.float)

            train_loss += 1000 * loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

            loss.backward()

            for ii, p in enumerate(model.state_dict()):
                if torch.all(torch.isnan(model.state_dict()[p])):
                    # model.init_weigths()
                    if epoch_counter > 0:
                        log_data = {'model': model,
                                    'trial': trial,
                                    'train_losses': train_losses,
                                    'test_losses': test_losses,
                                    'variance_list': variance_list,
                                    'to_minimize': to_minimize,
                                    'minimization_counter': minimization_counter,
                                    'error_thru_time': smallest_error_thru_time,
                                    'train_data_shape': train_shape,
                                    'test_data_shape': test_shape}
                        excel(study_path, 'update', log_data)
                        error_data_path = os.path.join(model_path, 'error_dataframe.pkl')
                        save(error_data_path, errors_dataframe)
                    if epoch_counter > 3:
                        loss_graph(variance_list, model_path, train_losses, test_losses, trial)
                    trial.report(smallest_error_thru_time, epoch_counter)
                    raise optuna.exceptions.TrialPruned()

            optimizer.step()
            del output, data, target
            torch.cuda.empty_cache()

        # Validation of the model.
        ground_truths, predictions = list(), list()
        with torch.no_grad():
            model.eval()
            all_test_output_data = []
            errors = []
            print('     Testing:')
            for x, ys, sample_indexes in tqdm(test_dataset):
                x = x.to(device)
                ys = ys.to(device)
                # forward
                predicted_ys = model(x)
                for predicted_y, true_y, sample_index in zip(predicted_ys, ys, sample_indexes):
                    loss = torch.sqrt(torch.sum(criterion(predicted_y, true_y)))
                    now_loss = loss.item()

                    target_cell = test_dataset.dataset.data_set[sample_index.item()].target_cell
                    time_range = test_dataset.dataset.data_set[sample_index.item()].time_range
                    all_test_output_data.append({'ground_truth': 1000 * true_y,
                                                 'prediction': 1000 * predicted_y,
                                                 'cell': target_cell, 'time_range': time_range,
                                                 'error': 1000 * now_loss})
                    errors.append(1000 * now_loss)

                # log
                predictions.append(predicted_ys.detach().cpu())
                del x, ys, predicted_ys
                torch.cuda.empty_cache()

        variance_list.append(calc_variance(predictions))

        error_thru_time = np.mean(errors)

        to_minimize = error_thru_time / (abs(len(train_dataset.dataset.allowed_times) / 10) ** 2)

        train_loss = train_loss / len(train_dataset)

        train_losses.append(train_loss)
        test_losses.append(error_thru_time)
        scheduler.step(error_thru_time)

        if smallest_error_thru_time > abs(error_thru_time):
            print('     Network improved, arranging data:')
            all_results = make_neat_epoch_data(all_test_output_data)
            errors_dataframe = make_an_error_data_frame(all_results)
            minimization_counter += 1
            smallest_error_thru_time = abs(error_thru_time)
            heatmap_values.append(delta_heatmap(errors_dataframe, heatmap_graph_path, epoch_counter, trial))
            trajectory_plot(all_results, trajectory_graph_path, trial, epoch_counter)
            error_thru_time_plot(errors_dataframe, error_thru_time_plot_path, trial, epoch_counter)
            error_data_path = os.path.join(model_path, 'error_dataframe - best.pkl')
            save(error_data_path, errors_dataframe)
            model_save_path = os.path.join(model_path, f'model - best.pt')
            torch.save(model, model_save_path)
            optimizer.param_groups[0]['lr'] = lr
            epochs = 20
            epoch = 0
            minimization_counter += 1

        current_samples = train_dataset.dataset.allowed_times

        train_dataset.dataset.allowed_samples(error_thru_time, 150, current_samples)
        test_dataset.dataset.allowed_samples(error_thru_time, 150, current_samples)
        if len(current_samples) < len(train_dataset.dataset.allowed_times):
            train_dataset.sampler.data_source = train_dataset.dataset.data_set
            test_dataset.sampler.data_source = test_dataset.dataset.data_set
            optimizer.param_groups[0]['lr'] = lr
            smallest_error_thru_time = 100000

        make_text_file(trial, model_path, train_losses, test_losses,
                       variance_list, model, general_settings, dataset_settings, train_shape,
                       test_shape)
        trial.report(to_minimize, epoch_counter)

        # Handle pruning based on the intermediate value.
        var_condition = (variance_list[-1]['variance'] < 1e-5 and epoch_counter > 20)
        # score_condition = (test_loss > 150 and epoch_counter >= 40)
        if trial.should_prune() or var_condition:  # or score_condition:
            if epoch_counter > 3:
                loss_graph(variance_list, model_path, train_losses, test_losses, trial)
                heatmap_values.append(delta_heatmap(errors_dataframe, heatmap_graph_path, epoch_counter, trial))
                heat_map_summary(heatmap_values, model_path)
                trajectory_plot(all_results, trajectory_graph_path, trial, f'Last ({epoch_counter})')
                error_thru_time_plot(errors_dataframe, error_thru_time_plot_path, trial, f'Last ({epoch_counter})')

                make_text_file(trial, model_path, train_losses, test_losses, variance_list, model,
                               general_settings, dataset_settings, train_shape, test_shape)
                model_save_path = os.path.join(model_path, f'model - last ({epoch_counter}).pt')
                torch.save(model, model_save_path)
                plt.close('all')

            log_data = {'model': model,
                        'trial': trial,
                        'train_losses': train_losses,
                        'test_losses': test_losses,
                        'variance_list': variance_list,
                        'to_minimize': to_minimize,
                        'minimization_counter': minimization_counter,
                        'error_thru_time': smallest_error_thru_time,
                        'train_data_shape': train_shape,
                        'test_data_shape': test_shape}

            excel(study_path, 'update', log_data)

            del (all_results, batch_idx, batch_size, best_trial_score, criterion, dataset_settings,
                 variance_list, epoch, error_thru_time_plot_path, general_settings, ground_truths, heatmap_graph_path,
                 heatmap_values, log_data, loss, loss_dict, lr, minimization_counter, model, model_path,
                 model_save_path, optimizer, predicted_y, predictions, sample_index,
                 sample_indexes, seed, test_dataset, test_losses, train_dataset, train_loss, train_losses,
                 trajectory_graph_path, true_y, weight_decay)
            torch.cuda.empty_cache()
            joblib.dump(study, os.path.join(study_path, study_name + '.pkl'))
             # delete_directory(model_path)
            raise optuna.exceptions.TrialPruned()

        epoch_counter += 1
        epoch += 1
        print(f'Current mean error: {error_thru_time:.4f}, best error: {smallest_error_thru_time:.4f}\n\n')

    loss_graph(variance_list, model_path, train_losses, test_losses, trial)
    heatmap_values.append(delta_heatmap(errors_dataframe, heatmap_graph_path, epoch_counter, trial))
    heat_map_summary(heatmap_values, model_path)
    trajectory_plot(all_results, trajectory_graph_path, trial, f'Last ({epoch_counter})')
    error_thru_time_plot(errors_dataframe, error_thru_time_plot_path, trial, f'Last ({epoch_counter})')

    make_text_file(trial, model_path, train_losses, test_losses, variance_list, model,
                   general_settings, dataset_settings, train_shape, test_shape)
    model_save_path = os.path.join(model_path, f'model - last ({epoch_counter}).pt')
    torch.save(model, model_save_path)
    plt.close('all')

    log_data = {'model': model,
                'trial': trial,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'variance_list': variance_list,
                'to_minimize': to_minimize,
                'minimization_counter': minimization_counter,
                'error_thru_time': smallest_error_thru_time,
                'train_data_shape': train_shape,
                'test_data_shape': test_shape}

    excel(study_path, 'update', log_data)

    del (all_results, batch_idx, batch_size, best_trial_score, criterion, dataset_settings,
         variance_list, epoch, error_thru_time_plot_path, general_settings, ground_truths, heatmap_graph_path,
         heatmap_values, log_data, loss, loss_dict, lr, minimization_counter, model, model_path,
         model_save_path, optimizer, predicted_y, predictions, sample_index,
         sample_indexes, seed, test_dataset, test_losses, train_dataset, train_loss, train_losses,
         trajectory_graph_path, true_y, weight_decay)
    torch.cuda.empty_cache()
    joblib.dump(study, os.path.join(study_path, study_name + '.pkl'))

    return to_minimize


if __name__ == "__main__":
    operating_system = 'linux'
    studies = 1
    files_to_copy = ['Optuna_LSTM_CNN.py', 'see_data_LSTM.py',
                     'Dataset_maker_LSTM.py', 'data_class_LSTM_CNN.py']
    datasets = ['train dataset - LSTM.pkl', 'test dataset - LSTM.pkl']
    dataset_type = input("What type of dataset is this? ")
    for _ in range(studies):
        study_num = str(int(time.time()))
        study_name = f'LSTM_CNN study - {study_num} - Nadav net'
        trial_start = f'Study id - {study_num} - Nadav net - {dataset_type}'

        if operating_system == 'windows':
            study_path = os.path.join(r'C:\Users\User\PycharmProjects\IMU\Raw data\models\LSTM', trial_start)
            dataset_graph_dir = r'C:\Users\User\PycharmProjects\IMU\Raw data\datasets\Preprocessed\figures'
            os.mkdir(study_path)
            dataset_graph_path = os.path.join(study_path, '---Dataset graphs')
            os.mkdir(dataset_graph_path)
        if operating_system == 'linux':
            study_path = os.path.join(r'/home/nadavk/IMU/Raw data/models/LSTM_CNN/', trial_start)
            work_dir = r'/home/nadavk/IMU/'
            dataset_graph_dir = r'/home/nadavk/IMU/Raw data/datasets/Preprocessed/figures/'
            os.mkdir(study_path)
            scripts_path = os.path.join(study_path, '---Scripts')
            os.mkdir(scripts_path)
            dataset_path = os.path.join(study_path, '---Datasets')
            os.mkdir(dataset_path)
            dataset_graph_path = os.path.join(study_path, '---Dataset graphs')
            os.mkdir(dataset_graph_path)

        copy_tree(dataset_graph_dir, dataset_graph_path)
        for file in files_to_copy:
            file_path = os.path.join(work_dir, file)
            copy(file_path, scripts_path)

        for file in datasets:
            file_path = os.path.join(r'/home/nadavk/IMU/Raw data/datasets/Preprocessed/', file)
            copy(file_path, dataset_path)

        excel(study_path, 'create', [])
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        DIR = os.getcwd()
        study_save_path = os.path.join(study_path, study_name + ' .pkl')

        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    study_name=study_name)
        study.optimize(objective, n_trials=1000)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
