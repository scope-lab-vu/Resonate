import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import scipy.optimize


def bin_and_avg_data(x, y, bins=None):
    if len(x) != len(y):
        raise ValueError("Input array size mismatch.")

    if bins is None:
        bins = [(0.0, 0.1),
                (0.1, 0.2),
                (0.2, 0.3),
                (0.3, 0.4),
                (0.4, 0.5),
                (0.5, 0.6),
                (0.6, 0.7),
                (0.7, 0.8),
                (0.8, 0.9),
                (0.9, 1.0),
                (1.0, 1.2),
                (1.2, 1.4),
                (1.4, 1.6),
                (1.6, 1.8),
                (1.8, 2.0),
                (2.0, 3.0),
                (3.0, 10.0)]

    avg_x, avg_y, std_dev_y = [], [], []
    for l_bound, u_bound in bins:
        bin_indices = np.nonzero(np.logical_and(x >= l_bound, x < u_bound))
        if np.size(bin_indices) > 0:
            binned_x = x[bin_indices]
            binned_y = y[bin_indices]
            avg_x.append(np.mean(binned_x))
            avg_y.append(np.mean(binned_y))
            std_dev_y.append(np.std(binned_y))

    return avg_x, avg_y, std_dev_y


def line_func(x, m, b):
    return m * x + b


def plot_results(file, save=False):
    # Setup plotting
    plt.rc('font', size=16)
    fig1 = plt.figure(dpi=300)
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.subplots_adjust(bottom=0.15)
    fig2 = plt.figure(dpi=300)
    ax2 = fig2.add_subplot(1, 1, 1)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    # Read results file
    df = pd.read_csv(file)
    est_freq = df["Avg_risk"].to_numpy()
    num_collisions = df["collisions"].to_numpy()
    # est_freq = np.tile(est_freq, 2)
    # num_collisions = np.tile(num_collisions, 2)

    # Plot num collisions vs average estimated frequency
    ax1.scatter(est_freq, num_collisions)
    ax1.set_xlabel("Average Estimated Frequency of Collisions")
    ax1.set_ylabel("Observed Number of Collisions")
    ax1.set_title("Observed vs. Estimated Collisions")
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim(-2, 20)
    ax1.text(0.1, 5.8, "n = %d" % len(est_freq))

    # Fit a trendline to the complete data set
    x_trend = np.array([0.0, 4.0], dtype='f')
    trendline_params = [1.0, 0.0]
    y_ideal = line_func(x_trend, *trendline_params)
    trendline_params, _ = scipy.optimize.curve_fit(line_func, est_freq, num_collisions, trendline_params)
    y_trend = line_func(x_trend, *trendline_params)

    # Fit a trendline to data points where x <= 1.0
    filtered_idx = np.nonzero(est_freq <= 1.0)
    filtered_x = est_freq[filtered_idx]
    filtered_y = num_collisions[filtered_idx]
    filtered_trendline_params, _ = scipy.optimize.curve_fit(line_func, filtered_x, filtered_y, trendline_params)
    y_trend_filtered = line_func(x_trend, *filtered_trendline_params)

    # Plot binned & averaged data and trend line
    binned_data = bin_and_avg_data(est_freq, num_collisions)
    ax2.scatter(binned_data[0], binned_data[1])
    ax2.plot(x_trend, y_trend, '--', color="orange")
    # ax2.plot(x_trend, y_ideal, '--', color="green")
    ax2.plot(x_trend, y_trend_filtered, ':', color="red")
    ax2.set_xlabel("Average Estimated Frequency of Collisions")
    ax2.set_ylabel("Observed Number of Collisions")
    ax2.set_title("Obs. vs. Est. Collisions (Binned)")
    ax2.text(2.05, 1.15, "%.3f * x + %.3f" % (trendline_params[0], trendline_params[1]))
    ax2.text(0.5, 2.05, "%.3f * x + %.3f" % (filtered_trendline_params[0], filtered_trendline_params[1]))
    # ax2.text(1.0, 1.8, "%.1f * x" % 1.0)
    ax2.set_ylim(0, np.ceil(np.max(y_trend)))
    ax2.set_xlim(0, np.ceil(np.max(x_trend)))
    ax2.grid(True)

    # Show plots and save to file if desired
    plt.show()
    if save:
        fig1.savefig("saved_images/results_scatterplot.pdf", dpi=300)
        fig2.savefig("saved_images/results_binned.pdf", dpi=300)

    # Calculate the average number of collisions per run
    avg_freq = np.sum(num_collisions) / float(len(num_collisions))
    print("Average Frequency: %f" % avg_freq)

    # Calculate overall likelihood, then likelihood in each range of values
    print("----- Overall ------")
    print("Count: %d" % len(est_freq))
    calc_likelihood(num_collisions, est_freq, avg_freq)
    for l_bound, u_bound in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        bounded_indices = np.nonzero(np.logical_and(est_freq >= l_bound, est_freq < u_bound))[0]
        bounded_collisions = num_collisions[bounded_indices]
        bounded_freq = est_freq[bounded_indices]
        print("----- [%d, %d) ------" % (l_bound, u_bound))
        print("Count: %d" % len(bounded_freq))
        calc_likelihood(bounded_collisions, bounded_freq, avg_freq)


def calc_likelihood(collisions, dynamic_freqs, static_freq):
    def poisson_log_likelihood(mu):
        return np.sum(np.log(scipy.stats.poisson.pmf(collisions, mu)))

    # Calculate likelihood using a poisson distribution
    dynamic_likelihood = poisson_log_likelihood(dynamic_freqs)
    static_likelihood = poisson_log_likelihood(static_freq)
    likelihood_ratio = dynamic_likelihood - static_likelihood
    print("Log Dynamic Likelihood: %f" % dynamic_likelihood)
    print("Log Static-Avg Likelihood: %f" % static_likelihood)
    print("Dynamic v. Static-Avg Likelihood Ratio: %f" % likelihood_ratio)

    # Use scipy to maximize log likelihood (by minimizing the negative) for a Poisson dist
    # guess = np.array([avg_freq])
    # max_likelihood_res = scipy.optimize.minimize(neg_poisson_log_likelihood, guess)
    # print(max_likelihood_res)


if __name__ == "__main__":
    RESULTS_FILE = "results_data/new-stats-v1.csv"
    plot_results(RESULTS_FILE, save=True)
