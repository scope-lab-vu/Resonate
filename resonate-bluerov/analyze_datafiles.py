import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
import datafile
import function_fitting
import multiprocessing as mp
import tqdm

AX1_DATA_LABELS = ["separation_dist", "far_encounter", "close_encounter", "collision"]
# AX2_DATA_LABELS = ["rgb_blur_percent"]
AX2_DATA_LABELS = []

# Dictionary for mapping data labels to more human-readable values, if desired
data_label_to_plot_label_map = {
    "separation_dist": "Separation Distance"
    # "monitor_result": "AM Log Martingale",
    # "rgb_blur": "Blur Detector",
    # "rgb_blur_percent": "Blur Metric"
}

AM_THRESHOLD = 20

# Parallelism options. Provides a speedup, but requires all objects to be pickle-able (bound class methods are not)
USE_MULTIPROCESSING = False  # Use Python multiprocessing.
NUM_WORKERS = 8  # If using multiprocessing, number of worker processes



def plot_data_file(df):
    # Setup plotting with two axes
    colors = list(mcolors.XKCD_COLORS.keys())
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    # Read all data labels of interest and plot on respective axis
    times = df.data["pose_gt"]["timestamp"]
    for data_label_set, axis in [(AX1_DATA_LABELS, ax1), (AX2_DATA_LABELS, ax2)]:
        for data_label in data_label_set:
            plot_label = data_label_to_plot_label_map.get(data_label, data_label)
            axis.plot(times, df.data[data_label][:], label=plot_label, color=colors.pop())

    # Plot additional metrics
    # ax1.plot(times, df.braking_margin, label="Braking Margin", color=colors.pop())
    # ax1.plot(times, datafile.am_moving_avg, label="AM LM Moving Avg.")

    # Plot a dotted zero-line for reference
    ax1.axhline(y=0, color='grey', linestyle="--")
    # ax1.plot(times, np.zeros(len(times)), '--', color=colors.pop())

    # Find and highlight the regions where threat, top event, or collision have occured
    highlight_region(df.data["far_encounter"], times, ax1, "green", "Threat")
    highlight_region(df.data["close_encounter"], times, ax1, "orange", "Top Event")
    highlight_region(df.data["collision"], times, ax1, "red", "Collision")

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
    # ax2.set_ylim(0, 1000)
    # ax1.legend(bbox_to_anchor=(-0.1, 1), loc='upper right', ncol=1)
    # ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    legend1 = ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    # Workaround to make matplotlib draw the ax1 legend properly (on top of ax2 plots)
    legend1.remove()
    ax2.add_artist(legend1)
    ax1.set_xlabel("Time (s)")
    ax1.set_title(df.filepath)
    plt.show()


def highlight_region(dataset, x_values, fig_ax, color, label):
    """Function to highlight regions of a figure where a binary dataset is True"""

    # Find all the transition points in the dataset (indices where data changes from True->False or vise-versa)
    transitions = np.flatnonzero(np.diff(dataset))
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


# def calculate_statistics(datafiles):
#     # Make enough cases for each fault mode plus any special cases
#     num_cases = len(fault_modes.SingularFaultModes) + 1
#     collision_count = np.zeros(num_cases)
#     top_event_count = np.zeros(num_cases)
#     trial_count = np.zeros(num_cases)
#     total_trials = 0
#     total_top = 0
#     total_collision = 0
#
#     # Calculate number of trials, collisions, and top events for each fault mode
#     for df in datafiles:
#         for fault in df.fault_set:
#             idx = fault.value
#             trial_count[idx] += 1
#             if df.collision_occurred:
#                 collision_count[idx] += 1
#             if df.top_occurred:
#                 top_event_count[idx] += 1
#
#             # Special case for no-fault scenarios
#             if fault == fault_modes.SingularFaultModes.NO_FAULT:
#                 if df.am_avg_before_top < AM_THRESHOLD:
#                     idx = num_cases - 1
#                     trial_count[idx] += 1
#                     if df.collision_occurred:
#                         collision_count[idx] += 1
#                     if df.top_occurred:
#                         top_event_count[idx] += 1
#
#         # Also keep a total count
#         total_trials += 1
#         if df.collision_occurred:
#             total_collision += 1
#         if df.top_occurred:
#             total_top += 1
#
#     # Print results
#     print("FM\t\tTrials\t\tTE\t\tTE%\t\tP|FM_EST\t\tP|!FM_EST\t\tC\t\tC%")
#     print("-----------------------------------------------------")
#     for fault in fault_modes.SingularFaultModes:
#         idx = fault.value
#         if trial_count[idx] > 0:
#             top_event_pct = 100 * top_event_count[idx]/float(trial_count[idx])
#             collision_pct = 100 * collision_count[idx] / float(trial_count[idx])
#         else:
#             top_event_pct = 0.0
#             collision_pct = 0.0
#         # Estimate probability of success with Laplace rule of succession
#         prob_success_est = (trial_count[idx] - top_event_count[idx] + 1)/float(trial_count[idx] + 2)
#         p_not_est = 1 - (total_top - top_event_count[idx] + 1)/float(total_trials - trial_count[idx] + 2)
#         print("%s\t%d\t%d\t%f\t%f\t%f\t%d\t%f" % (fault.name, trial_count[idx], top_event_count[idx], top_event_pct,
#                                               prob_success_est, p_not_est, collision_count[idx], collision_pct))
#
#         # Special case for no-fault scenarios
#         if fault == fault_modes.SingularFaultModes.NO_FAULT:
#             idx = num_cases - 1
#             if trial_count[idx] > 0:
#                 top_event_pct = 100 * top_event_count[idx] / float(trial_count[idx])
#                 collision_pct = 100 * collision_count[idx] / float(trial_count[idx])
#             else:
#                 top_event_pct = 0.0
#                 collision_pct = 0.0
#             # Estimate probability of success with Laplace rule of succession
#             prob_success_est = (trial_count[idx] - top_event_count[idx] + 1) / float(trial_count[idx] + 2)
#             print("%s\t%d\t%d\t%f\t%f\t%d\t%f" % ("NO_FAULT_LOW_AM", trial_count[idx], top_event_count[idx], top_event_pct,
#                                                   prob_success_est, collision_count[idx], collision_pct))
#
#     # Print totals (excluding the no-fault special case) as well
#     print("-----------------------------------------------------")
#     total_top_pct = 100 * total_top / total_trials
#     total_collision_pct = 100 * total_collision / total_trials
#     total_prob_success = (total_trials - total_top + 1) / float(total_trials + 2)
#     print("%s\t%d\t%d\t%f\t%f\t%d\t%f" % ("TOTALS", total_trials, total_top, total_top_pct,
#                                           total_prob_success, total_collision, total_collision_pct))


def plot_outcomes(datafiles):
    """Plot the outcome of each scenario against the average assurance monitor value"""

    # Compile outcomes of each sim run
    top_occurred = np.zeros(len(datafiles))
    collision_occurred = np.zeros(len(datafiles))
    thruster_degradation = np.zeros(len(datafiles))
    closest_approach = np.zeros(len(datafiles))
    for i, df in enumerate(datafiles):
        top_occurred[i] = df.top_occurred.astype(int)
        collision_occurred[i] = df.consequence_occurred.astype(int)
        thruster_degradation[i] = df.thruster_degradation_amount
        closest_approach[i] = df.closest_approach

    # Datasets with no top events or collisions can cause issues with max likelihood estimation. Print warning.
    if np.sum(top_occurred) == 0:
        print("WARNING: No Top events occurred in the provided datasets.")
    if np.sum(collision_occurred) == 0:
        print("WARNING: No Collision events occurred in the provided datasets.")

    # Calculate some simple statistics
    avg_top_prob = ((np.sum(top_occurred) + 1) / float(len(top_occurred) + 2))
    avg_col_prob = ((np.sum(collision_occurred) + 1) / float(len(collision_occurred) + 2))
    avg_col_prob_after_top = ((np.sum(collision_occurred) + 1) / float(np.sum(top_occurred) + 2))
    avg_ca = np.sum(closest_approach) / float(len(closest_approach))
    print("Average TOP Probability: %f" % avg_top_prob)
    print("Average Collision Probability: %f" % avg_col_prob)
    print("Average Collision Probability given TOP occurred: %f" % avg_col_prob_after_top)
    print("Average closest approach (m): %f" % avg_ca)
    print("\n\n")

    # Fit a conditional binomial distribution to the data with max-likelihood for the TOP event
    x_range = np.linspace(np.min(thruster_degradation), np.max(thruster_degradation), 100)
    fit_results = None
    try:
        fit_results, _ = function_fitting.max_likelihood_conditional_fit(thruster_degradation, top_occurred)
    except ValueError as e:
        print("Exception occurred when fitting sigmoid for TOP event:\n%s" % str(e))
    else:
        y_sig_max_likelihood = function_fitting.bounded_sigmoid(x_range, *fit_results.x)
        adjusted_likelihood = y_sig_max_likelihood/avg_top_prob
        print("TOP Sigmoid Fit: ", fit_results)
        print("\n\n")

    # Divide results into bins and average each bin
    bins = {"x": [], "y": {"top": [], "col": []}}
    bins_avg = {"x": [], "y": {"top": [], "col": []}}
    step_size = 0.2
    allowed_skew = 0.01
    for i in range(4):
        bin_x = step_size * i
        indices = np.logical_and(thruster_degradation > bin_x - allowed_skew, thruster_degradation < bin_x + allowed_skew)
        bins["x"].append(thruster_degradation[indices])
        bins["y"]["top"].append(top_occurred[indices])
        bins["y"]["col"].append(collision_occurred[indices])
        avg_deg = np.average(thruster_degradation[indices])
        avg_top_prob = (np.sum(top_occurred[indices]) + 1) / float(len(top_occurred[indices]) + 2)
        avg_col_prob = (np.sum(collision_occurred[indices]) + 1) / float(len(collision_occurred[indices]) + 2)
        bins_avg["x"].append(avg_deg)
        bins_avg["y"]["top"].append(avg_top_prob)
        bins_avg["y"]["col"].append(avg_col_prob)
        print("Bin %.2f contains %d entries.\n\tAverage probability of TOP: %.2f\n\tAverage probability of Collision: %.2f"
              % (avg_deg, len(thruster_degradation[indices]), avg_top_prob, avg_col_prob))
        print("\n\n")

    # Plot conditional probability P(B | D) and adjusted probability P(B | D) / P(B) for TOP barrier
    fig1 = plt.figure(dpi=300)
    ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.scatter(thruster_degradation, top_occurred, label="Top Events")
    # ax1.scatter(thruster_degradation, collision_occurred, label="Collision Events")
    ax1.scatter(bins_avg["x"], bins_avg["y"]["top"], label="Top Event Prob.")
    ax1.scatter(np.average(thruster_degradation), avg_top_prob, label="Average")
    if fit_results is not None:
        ax1.plot(x_range, y_sig_max_likelihood, label="Sigmoid MLE fit")
        ax1.plot(x_range, adjusted_likelihood, label="Adjusted fit")
        ax1.set_ylim([0.0, np.max(adjusted_likelihood)])
    else:
        ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel("Thruster Degradation Percentage")
    ax1.set_ylabel("TOP Event Probability")
    ax1.legend(loc="best")


    # Fit a conditional binomial distribution to the data with max-likelihood for the TOP event
    fit_results = None
    try:
        fit_results, _ = function_fitting.max_likelihood_conditional_fit(thruster_degradation, collision_occurred)
    except ValueError as e:
        print("Exception occurred when fitting sigmoid for Collision event:\n%s" % str(e))
    else:
        collision_y_sigmoid = function_fitting.bounded_sigmoid(x_range, *fit_results.x)
        collision_adj_likelihood = collision_y_sigmoid / avg_col_prob
        print("Collision Sigmoid Fit: ", fit_results)
        print("\n\n")

    # Plot conditional probability P(B | D) and adjusted probability P(B | D) / P(B) for Collision barrier
    fig2 = plt.figure(dpi=300)
    ax2 = fig2.add_subplot(1, 1, 1)
    # ax2.scatter(thruster_degradation, top_occurred, label="Top Events")
    # ax2.scatter(thruster_degradation, collision_occurred, label="Collision Events")
    ax2.scatter(bins_avg["x"], bins_avg["y"]["col"], label="Collision Event Prob.")
    ax2.scatter(np.average(thruster_degradation), avg_col_prob, label="Average")
    if fit_results is not None:
        ax2.plot(x_range, collision_y_sigmoid, label="Sigmoid MLE fit")
        ax2.plot(x_range, collision_adj_likelihood, label="Adjusted fit")
        ax2.set_ylim([0.0, np.max(collision_adj_likelihood)])
    else:
        ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("Thruster Degradation Percentage")
    ax2.set_ylabel("Collision Probability")
    ax2.legend(loc="best")

    # Show plots
    plt.show()


def read_datafile(file_path):
    df = datafile.DataFile(file_path)
    return df


if __name__ == "__main__":
    # DATA_FILE = "example_data/no_fault.csv"
    # DATA_FILE = "estimation_data/c_0.0_p_0.0_d_0.0/fm0.csv"
    # data_files = [DATA_FILE]

    DATA_DIR = "estimation_data"
    # DATA_DIR = "estimation_data/No-Faults/static/run3"

    # # Data with faults but without thruster reallocation
    # data_file_names = glob.glob(os.path.join(DATA_DIR, "**", "*.bag"), recursive=True)
    # data_file_names = [x for x in data_file_names if "/ais/" not in x]
    # data_file_names = [x for x in data_file_names if "/Fault-Reallocation/" not in x]
    # data_file_names = [x for x in data_file_names if "/Emergency-brake/" not in x]

    # # Data with faults AND thruster reallocation
    # data_file_names = glob.glob(os.path.join(DATA_DIR, "Fault-Reallocation/th0/**", "*.bag"), recursive=True)

    # Emergency Stop data
    # data_file_names = glob.glob(os.path.join(DATA_DIR, "No-Faults/static/Emergency-brake/**", "*.bag"), recursive=True)
    data_file_names = glob.glob(os.path.join(DATA_DIR, "ebrake/**", "*.bag"), recursive=True)

    # Read Datafiles
    datafiles = []
    if USE_MULTIPROCESSING:
        # Use multiple processes for parallelism.
        with mp.Pool(NUM_WORKERS) as pool:
            # Start the read operations asynchronously
            completion_count = 0
            for df in pool.imap_unordered(read_datafile, data_file_names):
                # Store data to dataset
                if df.data is not None:
                    datafiles.append(df)

                # Print update every 10 bag files
                completion_count += 1
                if completion_count % 10 == 0:
                    print("Processed %d/%d data files..." % (completion_count, len(data_file_names)))
    else:
        for filename in tqdm.tqdm(data_file_names):
            datafiles.append(read_datafile(filename))
    print("Finished processing data files.")

    # # Plot DataFile objects
    # for df in datafiles:
    #     df.plot_data()
    #     # plot_data_file(df)

    # calculate_statistics(datafiles)
    print("\n")

    plot_outcomes(datafiles)
