import matplotlib.pyplot as plt
import random
import scipy.optimize
import numpy as np
import math
from sigmoid import bounded_sigmoid


def curve_fit(x, y):
    p0 = [np.median(x), 1]
    return scipy.optimize.curve_fit(bounded_sigmoid, x, y, p0, method='dogbox')


def max_likelihood_conditional_fit(x, y):
    # Log Likelihood for binomial distribution with sigmoid conditional relationship (negative)
    def neg_sig_log_likelihood(theta):
        x0 = theta[0]
        k = theta[1]
        return -np.sum(y * np.log(bounded_sigmoid(x, x0, k)) + (1 - y) * np.log(1 - bounded_sigmoid(x, x0, k)))

    # Log Likelihood for binomial distribution with no conditional dependence
    def binomial_log_likelihood(p):
        successes = np.count_nonzero(y)
        trials = len(y)
        log_likelihood = math.log(p, math.e) * successes + math.log(1 - p, math.e) * (trials - successes)
        return log_likelihood

    # Use scipy to maximize log likelihood (by minimizing the negative) for various functions
    guess = np.array([np.median(x), 1.0])
    sig_max_likelihood_res = scipy.optimize.minimize(neg_sig_log_likelihood, guess)

    # Consider a sigmoid with the opposite conditional behavior as a sanity check. Likelihood should be extremely low.
    opposite_sig_max_likelihood_coeff = [1, -1] * sig_max_likelihood_res.x
    opposite_sig_max_likelihood = -neg_sig_log_likelihood(opposite_sig_max_likelihood_coeff)

    # Consider likelihood of an unconditional binomial
    p_value_sampled = np.count_nonzero(y) / len(y)
    unconditional_likelihood = binomial_log_likelihood(p_value_sampled)

    # Print likelihood results
    print("Log likelihood results:")
    print("Sigmoid: ", -sig_max_likelihood_res.fun)
    print("Opposite sigmoid: ", opposite_sig_max_likelihood)
    print("Unconditional: ", unconditional_likelihood)

    return sig_max_likelihood_res, opposite_sig_max_likelihood_coeff


if __name__ == "__main__":
    # Define true distribution function
    x_vals = np.linspace(-5, 21, 50)
    true_dist_params = np.array([6.0, 0.5])
    y_true = bounded_sigmoid(x_vals, *true_dist_params)

    # Randomly generate some example data and sort into bins
    x_sampled = []
    y_sampled = []
    bins = []
    martingale_centers = []
    starting_martingale = -4
    martingale_range_step = 2
    samples_per_range = 15
    sampled_num_successes = 0
    for i in range(13):
        lower_bound = starting_martingale + martingale_range_step * i
        upper_bound = starting_martingale + martingale_range_step * (i + 1)
        martingale_centers.append(float(lower_bound + upper_bound) / 2)
        samples = {"martingales": [], "outcomes": []}
        for j in range(samples_per_range):
            martingale = random.uniform(lower_bound, upper_bound)
            expected_prob = bounded_sigmoid(martingale, *true_dist_params)
            random_val = random.random()
            if random_val <= expected_prob:
                outcome = 1
                sampled_num_successes += 1
            else:
                outcome = 0
            x_sampled.append(martingale)
            y_sampled.append(outcome)
            samples["martingales"].append(martingale)
            samples["outcomes"].append(outcome)
        bins.append(samples)
    p_value_sampled = sampled_num_successes / len(y_sampled)
    x_sampled = np.array(x_sampled)
    y_sampled = np.array(y_sampled)

    # Plot probability for each bin
    average_outcomes = []
    for samples in bins:
        num_samples = len(samples["outcomes"])
        num_successes = 0
        for outcome in samples["outcomes"]:
            num_successes += outcome
        average_outcomes.append(num_successes / float(num_samples))
    # plt.scatter(martingale_centers, average_outcomes)
    # plt.xlabel("Martingale")
    # plt.ylabel("Average Outcome")
    # plt.xticks(martingale_centers)
    # plt.show()

    # Fit sigmoid to data with standard curve fitting
    raw_est = curve_fit(x_sampled, y_sampled)
    binned_est = curve_fit(martingale_centers, average_outcomes)
    y_raw_est = bounded_sigmoid(x_vals, *raw_est[0])
    y_binned_est = bounded_sigmoid(x_vals, *binned_est[0])
    print("Raw estimates:")
    print("popt: ", raw_est[0])
    print("pcov: ", raw_est[1])
    print("\n")
    # print("Binned estimates:")
    # print("popt: ", binned_est[0])
    # print("pcov: ", binned_est[1])
    # print("\n")

    # Fit sigmoid to data with Max Likelihood Estimation
    sig_max_likelihood_res, opposite_sig_max_likelihood_coeff = max_likelihood_conditional_fit(x_sampled, y_sampled)
    y_sig_max_likelihood = bounded_sigmoid(x_vals, *sig_max_likelihood_res.x)
    y_opposite_sig_max_likelihood = bounded_sigmoid(x_vals, *opposite_sig_max_likelihood_coeff)

    # Plot results
    # plt.plot(martingale_centers, average_outcomes, 'o', label='binned data')
    plt.plot(x_vals, y_true, label='true dist')
    plt.plot(x_sampled, y_sampled, 'o', label='raw data')
    plt.plot(x_vals, y_raw_est, label='raw sigmoid fit')
    # plt.plot(x_vals, y_binned_est, label='binned fit')
    plt.plot(x_vals, y_sig_max_likelihood, label='ML sigmoid fit')
    # plt.plot(x_vals, y_opposite_sig_max_likelihood, label='opposite max likelihood')
    plt.legend(loc='best')
    plt.xlabel("Log Martingale")
    plt.ylabel("P(B1 | S)")
    plt.show()