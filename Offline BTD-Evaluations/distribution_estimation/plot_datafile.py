import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
import datafile
import function_fitting
from risk_calculation import fault_modes

AX1_DATA_LABELS = ["Ground Truth", "stopping_distance", "monitor_result", "rgb_blur"]
AX2_DATA_LABELS = ["rgb_blur_percent"]

# Dictionary for mapping data labels to more human-readable values, if desired
data_label_to_plot_label_map = {
    "monitor_result": "AM Log Martingale",
    "stopping_distance": "Stopping Distance",
    "rgb_blur": "Blur Detector",
    "rgb_blur_percent": "Blur Metric"
}

AM_THRESHOLD = 20


def plot_data_file(df):
    # Setup plotting with two axes
    colors = list(mcolors.XKCD_COLORS.keys())
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    # Read all data labels of interest and plot on respective axis
    times = df.times
    for data_label_set, axis in [(AX1_DATA_LABELS, ax1), (AX2_DATA_LABELS, ax2)]:
        for data_label in data_label_set:
            plot_label = data_label_to_plot_label_map.get(data_label, data_label)
            axis.plot(times, df.data[data_label][:], label=plot_label, color=colors.pop())

    # Plot additional metrics
    ax1.plot(times, df.braking_margin, label="Braking Margin", color=colors.pop())
    # ax1.plot(times, datafile.am_moving_avg, label="AM LM Moving Avg.")

    # Plot a dotted zero-line for reference
    ax1.axhline(y=0, color='grey', linestyle="--")
    # ax1.plot(times, np.zeros(len(times)), '--', color=colors.pop())

    # Find and highlight the regions where braking margin is below desired minimum
    highlight_region(df.braking_threshold_violations, times, ax1, "orange", "Top Event")

    # Also want to highlight regions where a collision has occurred
    highlight_region(df.collisions, times, ax1, "red", "Collision")

    # Calculate closing speed and Time-To-Collision (TTC) values
    # closing_speeds = []
    # ttcs = []
    # last_sep_dist = datafile.data["Ground Truth"][0]
    # for cur_sep_dist in datafile.data["Ground Truth"][1:]:
    #     delta_sep_dist = last_sep_dist - cur_sep_dist
    #     closing_speed = delta_sep_dist / STEP_DELTA_T_S
    #     closing_speeds.append(closing_speed)
    #     if abs(closing_speed) > 0.2:
    #         time_to_collision = cur_sep_dist / closing_speed
    #         ttcs.append(max(min(time_to_collision, 100), -10))
    #     else:
    #         ttcs.append(0)
    #     last_sep_dist = cur_sep_dist
    # plt.plot(ttcs, label="Time to Collision")
    # plt.plot(closing_speeds, label="closing speed")

    # Place legends, label axes, and set limits as desired
    # ax1.set_xlim(15, 20)
    # ax1.set_ylim(-2, 20)
    ax2.set_ylim(0, 1000)
    # ax1.legend(bbox_to_anchor=(-0.1, 1), loc='upper right', ncol=1)
    # ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    legend1 = ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    # Workaround to make matplotlib draw the ax1 legend properly (on top of ax2 plots)
    legend1.remove()
    ax2.add_artist(legend1)
    ax1.set_xlabel("Time (s)")
    ax1.set_title(df.filename)
    plt.show()


def highlight_region(dataset, x_values, fig_ax, color, label):
    """Function to highlight regions of a figure where a binary dataset is True"""

    # Find all the transition points in the dataset (indices where data changes from True->False or vise-versa)
    transitions = np.flatnonzero(dataset.diff().to_numpy())
    highlight_exists = False
    for i, transition_index in enumerate(transitions):
        # Only want to shade regions where original dataset is True
        if dataset[transition_index]:
            # Find end of this region by finding the next index where a transition occurs
            if i + 1 < len(transitions):
                next_transition_index = transitions[i + 1]
            else:
                next_transition_index = len(dataset) - 1

            # Translate transition indices to x-values
            start_x = x_values[transition_index]
            end_x = x_values[next_transition_index]

            # Highlight region in plot, but only label once to avoid duplicate labels
            if highlight_exists:
                fig_ax.axvspan(start_x, end_x, color=color, alpha=0.2)
            else:
                fig_ax.axvspan(start_x, end_x, color=color, label=label, alpha=0.2)
                highlight_exists = True


def calculate_statistics(datafiles):
    # Make enough cases for each fault mode plus any special cases
    num_cases = len(fault_modes.SingularFaultModes) + 1
    collision_count = np.zeros(num_cases)
    top_event_count = np.zeros(num_cases)
    trial_count = np.zeros(num_cases)
    total_trials = 0
    total_top = 0
    total_collision = 0

    # Calculate number of trials, collisions, and top events for each fault mode
    for df in datafiles:
        for fault in df.fault_set:
            idx = fault.value
            trial_count[idx] += 1
            if df.collision_occurred:
                collision_count[idx] += 1
            if df.top_occurred:
                top_event_count[idx] += 1

            # Special case for no-fault scenarios
            if fault == fault_modes.SingularFaultModes.NO_FAULT:
                if df.am_avg_before_top < AM_THRESHOLD:
                    idx = num_cases - 1
                    trial_count[idx] += 1
                    if df.collision_occurred:
                        collision_count[idx] += 1
                    if df.top_occurred:
                        top_event_count[idx] += 1

        # Also keep a total count
        total_trials += 1
        if df.collision_occurred:
            total_collision += 1
        if df.top_occurred:
            total_top += 1

    # Print results
    print("FM\t\tTrials\t\tTE\t\tTE%\t\tP|FM_EST\t\tP|!FM_EST\t\tC\t\tC%")
    print("-----------------------------------------------------")
    for fault in fault_modes.SingularFaultModes:
        idx = fault.value
        if trial_count[idx] > 0:
            top_event_pct = 100 * top_event_count[idx]/float(trial_count[idx])
            collision_pct = 100 * collision_count[idx] / float(trial_count[idx])
        else:
            top_event_pct = 0.0
            collision_pct = 0.0
        # Estimate probability of success with Laplace rule of succession
        prob_success_est = (trial_count[idx] - top_event_count[idx] + 1)/float(trial_count[idx] + 2)
        p_not_est = 1 - (total_top - top_event_count[idx] + 1)/float(total_trials - trial_count[idx] + 2)
        print("%s\t%d\t%d\t%f\t%f\t%f\t%d\t%f" % (fault.name, trial_count[idx], top_event_count[idx], top_event_pct,
                                              prob_success_est, p_not_est, collision_count[idx], collision_pct))

        # Special case for no-fault scenarios
        if fault == fault_modes.SingularFaultModes.NO_FAULT:
            idx = num_cases - 1
            if trial_count[idx] > 0:
                top_event_pct = 100 * top_event_count[idx] / float(trial_count[idx])
                collision_pct = 100 * collision_count[idx] / float(trial_count[idx])
            else:
                top_event_pct = 0.0
                collision_pct = 0.0
            # Estimate probability of success with Laplace rule of succession
            prob_success_est = (trial_count[idx] - top_event_count[idx] + 1) / float(trial_count[idx] + 2)
            print("%s\t%d\t%d\t%f\t%f\t%d\t%f" % ("NO_FAULT_LOW_AM", trial_count[idx], top_event_count[idx], top_event_pct,
                                                  prob_success_est, collision_count[idx], collision_pct))

    # Print totals (excluding the no-fault special case) as well
    print("-----------------------------------------------------")
    total_top_pct = 100 * total_top / total_trials
    total_collision_pct = 100 * total_collision / total_trials
    total_prob_success = (total_trials - total_top + 1) / float(total_trials + 2)
    print("%s\t%d\t%d\t%f\t%f\t%d\t%f" % ("TOTALS", total_trials, total_top, total_top_pct,
                                          total_prob_success, total_collision, total_collision_pct))


def plot_outcomes(datafiles):
    """Plot the outcome of each scenario against the average assurance monitor value"""

    # Compile AM values and outcomes of all data files with NO FAULT condition
    no_fault_dfs = []
    for df in datafiles:
        # if len(df.fault_set) == 0 or fault_modes.SingularFaultModes.NO_FAULT in df.fault_set:
        #     no_fault_dfs.append(df)
        no_fault_dfs.append(df)
    top_occurred = np.zeros(len(no_fault_dfs))
    collision_occurred = np.zeros(len(no_fault_dfs))
    average_am = np.zeros(len(no_fault_dfs))
    for i, df in enumerate(no_fault_dfs):
        top_occurred[i] = df.top_occurred
        collision_occurred[i] = df.collision_occurred
        average_am[i] = df.am_avg_before_top
    avg_prob = ((np.sum(top_occurred) + 1) / float(len(top_occurred) + 2))
    print("Average TOP Probability: %f" % avg_prob)

    # Fit a conditional binomial distribution to the data with max-likelihood
    fit_results, _ = function_fitting.max_likelihood_conditional_fit(average_am, top_occurred)
    x_range = np.linspace(np.min(average_am), np.max(average_am), 100)
    y_sig_max_likelihood = function_fitting.bounded_sigmoid(x_range, *fit_results.x)
    print("Sigmoid Fit: ", fit_results)

    # Plot collisions and top events on same figure
    # Plot conditional probability P(B | AM) and adjusted probability P(B | AM) / P(B)
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(average_am, top_occurred, label="Top Events")
    # ax1.scatter(average_am, collision_occurred, label="Collision Events")
    ax1.plot(x_range, y_sig_max_likelihood, label="Sigmoid MLE fit")
    ax1.plot(x_range, y_sig_max_likelihood/avg_prob, label="Adjusted fit")
    ax1.set_xlabel("Average AM Log Martingale")
    ax1.set_ylabel("Outcome (True/False)")
    ax1.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    # DATA_FILE = "example_data/no_fault.csv"
    # DATA_FILE = "estimation_data/c_0.0_p_0.0_d_0.0/fm0.csv"
    # data_files = [DATA_FILE]

    DATA_DIR = "estimation_data"
    data_file_names = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)

    # Construct DataFile objects
    datafiles = []
    for file in data_file_names:
        df = datafile.DataFile()
        df.load_csv(file)
        df.calc_metrics()
        datafiles.append(df)

    # # Plot each datafile
    # for df in datafiles:
    #     plot_data_file(df)

    calculate_statistics(datafiles)
    print("\n")

    plot_outcomes(datafiles)
