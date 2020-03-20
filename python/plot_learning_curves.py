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

if __name__ == "__main__":
    matplotlib.rc('font', size=16)
    parser = argparse.ArgumentParser("Function for plotting learning curves for saved logs of RL rollouts.")
    parser.add_argument("-ld", "--log_dirs", nargs="+", help="<required> Set flag", required=True)
    parser.add_argument("-rn", "--run_names", nargs="+", help="<required> Set flag", required=True)
    parser.add_argument("-c", "--colors", nargs="+", help="<required> Set flag", required=True, default=["b", "g", "r", "c", "m", "y", "k"])
    parser.add_argument("-t", "--title", help="title of plot", required=False, default="Learning Curves")
    parser.add_argument("-mean_var", help="mean and variance plots", required=False, default=None)
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
        else:
            compare_results(log_folders=log_dirs, names=names, title=title, colors=colors)

# if __name__ == "__main__":
#     try:
#         log_dir_index = -1
#         run_name_index = -1
#         colors_index = -1
#         for i in range(len(sys.argv)):
#             # print(sys.argv[i][0:len('log_dirs=')])
#             # print(sys.argv[i][0:len('run_names=')])
#             # print(sys.argv[i][0:len('colors=')])
#             # print(type(sys.argv[i][0:len('colors=')]))
#
#             if sys.argv[i][0:len('log_dirs=')] == 'log_dirs=':
#                 log_dir_index = i
#             if sys.argv[i][0:len('run_names=')] == 'run_names=':
#                 run_name_index = i
#             if sys.argv[i][0:len('colors=')] == 'colors=':
#                 colors_index = i
#
#         # print(log_dir_index)
#         # print(run_name_index)
#         # print(colors_index)
#
#         if log_dir_index == -1 or run_name_index == -1:
#             raise StringException("Usage: log_dirs=<comma separated log dirs> run_names=<comma separated run names> colors=<comma separated color initials (optional)>")
#
#
#         log_dirs = sys.argv[log_dir_index][len('log_dirs='):].split(',')
#         names = sys.argv[run_name_index][len('run_names='):].split(',')
#         if colors_index != -1:
#             colors = sys.argv[colors_index][len('colors='):].split(',')
#         else:
#             colors = None
#
#         if len(log_dirs) != len(names) or len(names) != len(colors) and colors is not None:
#             raise StringException("Length of lists must be equal.\n" + str(log_dirs) + "\n" + str(names) + "\n" + str(colors))
#
#         top_dir = "training_logs/"
#         for i in range(len(log_dirs)):
#             log_dirs[i] = top_dir + log_dirs[i]
#             names[i] = names[i].replace('_',' ')
#         print("colors:",colors)
#         if colors is not None:
#             compare_results(log_folders=log_dirs, names=names, colors=colors)
#         else:
#             compare_results(log_folders=log_dirs, names=names)
#
#     except StringException as e:
#         print(e.what())

# if __name__ == "__main__":
#     try:
#         if len(sys.argv) < 2:
#             raise StringException("Usage: training_logs/<list of log directory/run-name pairs>")
#
#         top_dir = "training_logs/"
#         log_dirs = []
#         names = []
#         for i in range(int(len(sys.argv)/2)):
#             log_dirs.append(top_dir + sys.argv[2 * i + 1])
#             names.append(sys.argv[2 * i + 2].replace('_',' '))
#         compare_results(log_dirs, names)
#
#     except StringException as e:
#         print(e.what())


# run_name = "A2C3_freespacetraj_noGravComp_g0"
# log_dir = "training_logs/" + run_name + "/"
# plot_results(log_dir, title="Free Space Trajectory: No Gravity Compensation")
# results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "ACKTR_freespacetraj")
