from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt
import sys
from helperFun import StringException


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

if __name__ == "__main__":
    try:
        log_dir_index = -1
        run_name_index = -1
        colors_index = -1
        for i in range(len(sys.argv)):
            # print(sys.argv[i][0:len('log_dirs=')])
            # print(sys.argv[i][0:len('run_names=')])
            # print(sys.argv[i][0:len('colors=')])
            # print(type(sys.argv[i][0:len('colors=')]))

            if sys.argv[i][0:len('log_dirs=')] == 'log_dirs=':
                log_dir_index = i
            if sys.argv[i][0:len('run_names=')] == 'run_names=':
                run_name_index = i
            if sys.argv[i][0:len('colors=')] == 'colors=':
                colors_index = i

        # print(log_dir_index)
        # print(run_name_index)
        # print(colors_index)

        if log_dir_index == -1 or run_name_index == -1:
            raise StringException("Usage: log_dirs=<comma separated log dirs> run_names=<comma separated run names> colors=<comma separated color initials (optional)>")


        log_dirs = sys.argv[log_dir_index][len('log_dirs='):].split(',')
        names = sys.argv[run_name_index][len('run_names='):].split(',')
        if colors_index != -1:
            colors = sys.argv[colors_index][len('colors='):].split(',')
        else:
            colors = None

        if len(log_dirs) != len(names) or len(names) != len(colors) and colors is not None:
            raise StringException("Length of lists must be equal.\n" + str(log_dirs) + "\n" + str(names) + "\n" + str(colors))

        top_dir = "training_logs/"
        for i in range(len(log_dirs)):
            log_dirs[i] = top_dir + log_dirs[i]
            names[i] = names[i].replace('_',' ')
        print("colors:",colors)
        if colors is not None:
            compare_results(log_folders=log_dirs, names=names, colors=colors)
        else:
            compare_results(log_folders=log_dirs, names=names)

    except StringException as e:
        print(e.what())

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
