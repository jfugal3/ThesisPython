from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from helperFun import StringException
import argparse
import os


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # x = x/(1e6)
    y_smooth = moving_average(y, window=50)
    # Truncate x
    x_smooth = x[len(x) - len(y_smooth):]

    fig = plt.figure(title)
    print(len(y_smooth))
    print(len(x))
    plt.plot(x_smooth, y_smooth, 'b-')
    plt.plot(x, y, 'b.', alpha=0.2)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()


def compare_results(log_folders, names, title="Learning Curves", colors=["b", "g", "r", "c", "m", "y", "k"]):
    fig = plt.figure()
    if len(log_folders) > 7 and len(log_folders) > len(colors):
        print(log_folders)
        print(colors)
        print("Thats too many plots!\nTry 7 or less. Or try specifying a color for each plot.")
        return

    used_colors=[]
    for i in range(len(log_folders)):
        x, y = ts2xy(load_results(log_folders[i]), 'timesteps')
        y_smooth = moving_average(y, window=50)
        x_smooth = x[len(x) - len(y_smooth):]
        plt.plot(x,y, colors[i]+".", alpha=0.2)
        if colors[i] not in used_colors:
            plt.plot(x_smooth,y_smooth, colors[i]+"-", label=names[i])
            used_colors.append(colors[i])
        else:
            plt.plot(x_smooth,y_smooth, colors[i]+"-")

    plt.legend()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

def mean_var_plots(log_folders, names, title="Learning Curves", colors=["b", "g", "r", "c", "m", "y", "k"]):
    fig = plt.figure()
    if len(log_folders) > 7 and len(log_folders) > len(colors):
        print(log_folders)
        print(colors)
        print("Thats too many plots!\nTry 7 or less. Or try specifying a color for each plot.")
        return

    used_colors=[]
    for i in range(len(log_folders)):
        y_mat = []
        for directory_name in os.listdir(log_folders[i]):
            if directory_name.endswith("monitor_dir"):
                x, y = ts2xy(load_results(os.path.join(log_folders[i], directory_name)), 'timesteps')
                y_mat.append(y)

        assert len(y_mat) != 0, "found no directories ending in 'monitor_dir' in " + log_folders[i]
        y_mat = np.vstack(y_mat)
        mean = np.mean(y_mat, axis=0)
        std = np.std(y_mat, axis=0)
        plt.plot(x, mean, colors[i]+"-", label=names[i])
        plt.fill_between(x, mean + std/2, mean - std/2, color=colors[i], alpha=0.3)

    plt.legend()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

def grid_analysis(log_dirs, names, sess_100_dirs=None, sess_names=None, colors=["b", "g", "r", "c", "m", "y", "k"]):
    fig = plt.figure()
    dir_num = 0
    for log_folder in log_dirs:
        log_folders = os.listdir(log_folder)
        x_arr = []
        y_arr = []
        e_arr = []
        h_arr = []
        for i in range(len(log_folders)):
            y_mat = []
            for directory_name in os.listdir(os.path.join(log_folder,log_folders[i])):
                if directory_name.endswith("monitor_dir"):
                    x, y = ts2xy(load_results(os.path.join(log_folder,log_folders[i], directory_name)), 'timesteps')
                    y_mat.append(y)
                    # print(len(x))

            assert len(y_mat) != 0, "found no directories ending in 'monitor_dir' in " + log_folders[i]
            y_mat = np.vstack(y_mat)
            y_mean = np.mean(y_mat)
            y_std = np.std(np.mean(y_mat,0))
            y_arr.append(y_mean)
            e_arr.append(y_std/2)
            h_arr.append(log_folders[i])

        yeh = zip(y_arr, e_arr, h_arr)
        y_sort = []
        x_sort = []
        e_sort = []
        h_sort = []
        i = 1 + dir_num * 0.5
        for y, e, h in reversed(sorted(yeh)):
            y_sort.append(y)
            x_sort.append(i)
            e_sort.append(e)
            h_sort.append(h)
            i += 1
        out_file_name = "output{}.txt".format(dir_num)
        f = open(out_file_name, "w")
        f.write(",")
        for key_val in h_sort[0].split("__"):
            f.write(key_val.split("-")[0] + ",")
        f.write("\n")
        i = 1
        for hyperparams in h_sort:
            f.write(str(i) + ",")
            for key_val in hyperparams.split("__"):
                f.write(str(key_val.split("-")[1]) + ",")
            f.write(str(np.around(y_sort[i-1],2)) + "," + str(np.around(e_sort[i-1] * 2, 2)) + ",")
            f.write("\n")
            i += 1
        f.close()
        max_mean_index = np.where(y_sort == np.amax(y_sort))[0][0]
        print("Writing output to " + out_file_name)
        plt.errorbar(x_sort, y_sort, e_sort, fmt='.', capsize=3, label=names[dir_num], color=colors[dir_num])
        # plt.errorbar(x_sort[max_mean_index], y_sort[max_mean_index], e_sort[max_mean_index], fmt='r.', capsize=3)
        dir_num += 1
    print("sess_100_dirs", sess_100_dirs)
    if sess_100_dirs is not None:
        sess_num = 0.0
        for i in range(len(sess_100_dirs)):
            y_mat = []
            for directory_name in os.listdir(sess_100_dirs[i]):
                if directory_name.endswith("monitor_dir"):
                    x, y = ts2xy(load_results(os.path.join(sess_100_dirs[i], directory_name)), 'timesteps')
                    y_mat.append(y)
                    # print(len(x))

            assert len(y_mat) != 0, "found no directories ending in 'monitor_dir' in " + log_folders[i]
            y_mat = np.vstack(y_mat)
            y_mean = np.mean(y_mat)
            y_std = np.std(np.mean(y_mat,0))
            sess_num += 0.5
            plt.errorbar(sess_num, y_mean, y_std / 2, fmt='.', capsize=3, label=sess_names[i])

    plt.legend()
    # plt.locator_params(nbins=4)
    plt.xlim([0, 193])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Hyper Parameter Permutation Ranking')
    plt.ylabel('Area Under the Learning Curve')
    plt.title('Grid Search Results')
    plt.show()

if __name__ == "__main__":
    matplotlib.rc('font', size=16)
    parser = argparse.ArgumentParser("Function for plotting learning curves for saved logs of RL rollouts.")
    parser.add_argument("-ld", "--log_dirs", nargs="+", help="<required> Set flag", required=True)
    parser.add_argument("-rn", "--run_names", nargs="+", help="<required> Set flag", required=True)
    parser.add_argument("-c", "--colors", nargs="+", help="<required> Set flag", required=False, default=["default"])
    parser.add_argument("-t", "--title", help="title of plot", required=False, default="Learning Curves")
    parser.add_argument("-mean_var", help="mean and variance plots", required=False, default=None)
    parser.add_argument("-grid_analysis", help="analyze grid search results", required=False, default=None)
    parser.add_argument("-sess_100_dirs", nargs="+", help="dirs for grid_analysis", required=False, default=None)
    parser.add_argument("-sess_names", nargs="+", help="names for sess_100_dirs", required=False, default=None)
    args = parser.parse_args()

    log_dirs = args.log_dirs
    names = args.run_names
    colors = args.colors
    title = args.title.replace('_', ' ')
    if colors[0] == "default":
        colors = ["b", "g", "r", "c", "m", "y", "k"][:len(names)]
    if len(log_dirs) != len(names) or len(log_dirs) != len(colors):
        print("number of log directories, run_names, and colors must be equal.")
        print("len(log_dirs) =", len(log_dirs))
        print("len(run_names) =", len(names))
        print("len(colors) =", len(colors))
    else:
        top_dir = "training_logs"
        for i in range(len(log_dirs)):
            log_dirs[i] = os.path.join(top_dir, log_dirs[i])
            names[i] = names[i].replace('_', ' ')
        if args.mean_var is not None:
            mean_var_plots(log_folders=log_dirs, names=names, title=title, colors=colors)
        elif args.grid_analysis is not None:
            if args.sess_100_dirs is not None:
                for i in range(len(args.sess_100_dirs)):
                    args.sess_100_dirs[i] = os.path.join(top_dir, args.sess_100_dirs[i])
            grid_analysis(log_dirs, names, args.sess_100_dirs, args.sess_names)
        else:
            compare_results(log_folders=log_dirs, names=names, title=title, colors=colors)
