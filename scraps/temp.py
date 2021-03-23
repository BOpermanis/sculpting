import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
N = 4
n_prog = 4
noise = np.random.normal(0, 2, (N, n_prog))
trend = np.clip(np.random.normal(0, 1, (n_prog, )), -np.inf, 0)# jo paarsvaraa katrai studiju programma nepieaugs
intercept = np.random.normal(30, 6, (n_prog, ))
noise[:, 1:] = np.clip(noise[:, 1:], -np.inf, 0) #

start_year = 2021

def plot_all_programs():
    labels = []
    handles = []
    for i in range(n_prog):
        years = start_year + np.arange(N)
        pupil_cnts = np.arange(N) * trend[i] + intercept[i] + noise[:, i]
        # line_up, = plt.plot([1, 2, 3], label='Line 2')
        # line_down, = plt.plot([3, 2, 1], label='Line 1')
        labels.append("Mācību programma " + str(i + 1))
        ln, = plt.plot(years, pupil_cnts, label=labels[-1])
        handles.append(ln)

    plt.legend(handles, labels)

    plt.show()

def plot_one_with_confidence_interval(i):
    years = start_year + np.arange(N)
    pupil_cnts = np.arange(N) * trend[i] + intercept[i] + noise[:, i]
    # plt.plot(years, pupil_cnts)
    # plt.show()
    fig, ax = plt.subplots()
    ax.plot(years, pupil_cnts, label="Mācību programma 1")
    c_lower = 2 + np.arange(N) ** 2
    c_upper = 1 + np.arange(N)
    ax.fill_between(years, (pupil_cnts - c_lower), (pupil_cnts + c_upper), alpha=.13, label="95% ticamības intervāls")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()

# plot_all_programs()
plot_one_with_confidence_interval(0)
