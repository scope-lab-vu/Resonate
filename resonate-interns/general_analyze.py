import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
import function_fitting
import tqdm
import concurrent.futures
from UUV_datafile import UUV_Datafile
from CARLA_datafile import CARLA_Datafile
import fault_modes
import argparse
from argparse import RawTextHelpFormatter


class CollatedData(object):
    def __init__(self):
        self.top_events = [] # list of booleans
        self.consequences = [] # list of booleans
        self.list_independent_vars_cont = {} # dictionary, string -> list of floats
        self.list_independent_vars_disc = {} # dictionary, string -> list of ints
        

    def build(self, list_of_data_objs):
        self.list_independent_vars_cont = {}
        for data in list_of_data_objs:
            self.top_events.append(data.top_event)
            self.consequences.append(data.consequence)
            if(data.independent_vars_cont):
                for var_name, var_value in data.independent_vars_cont.items():
                    if self.list_independent_vars_cont.get(var_name, False):
                        self.list_independent_vars_cont[var_name].append(var_value)
                    else:
                        self.list_independent_vars_cont[var_name] = [var_value]
            if(data.independent_vars_disc):
                for var_name, var_value in data.independent_vars_disc.items():
                    if self.list_independent_vars_disc.get(var_name, False):
                        self.list_independent_vars_disc[var_name].append(var_value)
                    else:
                        self.list_independent_vars_disc[var_name] = [var_value]

    def get_run_info(self, run_index):
        dt = Datafile()
        dt.top_event = self.top_events[run_index]
        dt.consequence = self.consequences[run_index]
        dt.independent_vars_cont = {}
        dt.independent_vars_disc = {}
        for key, value in self.list_independent_vars_cont.items():
            dt.independent_vars_cont[key] = value[run_index]
        for key, value in self.list_independent_vars_disc.items():
            dt.independent_vars_disc[key] = value[run_index]
        return dt

    def show(self):
        print("Top events:", self.top_events)
        print("Consequences:", self.consequences)
        for key, value in self.list_independent_vars_cont.items():
            print("{}: {}".format(key, value))
        for key, value in self.list_independent_vars_disc.items():
            print("{}: {}".format(key, value))
        print('')

# Produces the stats and table for all disc. variables
def disc_anaylsis(collated_data):
    num_cases = len(collated_data.list_independent_vars_disc) + 1
    consequence_count = np.zeros(num_cases)
    top_event_count = np.zeros(num_cases)
    trial_count = np.zeros(num_cases) # array of the size of indep var, each index would be a different var
    total_trials = 0                    # or even better, make an object
    total_top = 0
    total_consequence = 0

    for trial in range(len(collated_data.top_events)):
        check_for_no_disc = 0
        for disc_var in range(num_cases - 1):
            if collated_data.list_independent_vars_disc.values()[disc_var][trial]: # check if the disc variable occured  
                check_for_no_disc += 1
                trial_count[disc_var] += 1 # disc var 1, i.e. the first one defined in the dict, will relate to the first index in the arrays.
                if(collated_data.consequences[trial]):
                    consequence_count[disc_var] += 1
                if(collated_data.top_events[trial]):
                    top_event_count[disc_var] += 1

        
        # at this point i should check to see if no disc vars happened in this trail
        # last index, i = num_cases-1, is reserved for no disc_var
        if(check_for_no_disc == 0): #i.e. no disc var occurred in this trial
            trial_count[num_cases-1] += 1 # disc var 1, i.e. the first one defined in the dict, will relate to the first index in the arrays.
            if(collated_data.consequences[trial]):
                consequence_count[num_cases-1] += 1
            if(collated_data.top_events[trial]):
                top_event_count[num_cases-1] += 1

        total_trials += 1
        if collated_data.consequences[trial]:
            total_consequence += 1
        if collated_data.top_events[trial]:
            total_top += 1


    # Print results  
    print("Variable\t\tTrials\tTE\tTE%\tP|FM_EST  P|!FM_EST\tC\tC%")
    print("---------------------------------------------------------------------------------------")

    list_disc_var_names = collated_data.list_independent_vars_disc.keys()
    # print(list_disc_var_names)
    
    for disc_var in range(num_cases): #todo make sure this range is correct
        if trial_count[disc_var] > 0:
            top_event_pct = 100 * top_event_count[disc_var]/float(trial_count[disc_var])
            consequence_pct = 100 * consequence_count[disc_var] / float(trial_count[disc_var])
        else:
            top_event_pct = 0.0
            consequence_pct = 0.0

        prob_success_est = (trial_count[disc_var] - top_event_count[disc_var] + 1)/float(trial_count[disc_var] + 2)    
        p_not_est = 1 - (total_top - top_event_count[disc_var] + 1)/float(total_trials - trial_count[disc_var] + 2)
        if disc_var != num_cases - 1:
            print("%.15s\t\t%d\t%d\t%.2f\t%.2f\t  %.2f  \t%d\t%.2f" % (list_disc_var_names[disc_var], trial_count[disc_var], top_event_count[disc_var], top_event_pct,
                                              prob_success_est, p_not_est, consequence_count[disc_var], consequence_pct))
        else:
            print("%.15s\t\t%d\t%d\t%.2f\t%.2f\t  %.2f  \t%d\t%.2f" % ("No disc var", trial_count[disc_var], top_event_count[disc_var], top_event_pct,
                                              prob_success_est, p_not_est, consequence_count[disc_var], consequence_pct))


    # Print totals (excluding the no-fault special case) as well
    print("---------------------------------------------------------------------------------------")
    total_top_pct = 100 * total_top / total_trials
    total_consequence_pct = 100 * total_consequence / total_trials
    total_prob_success = (total_trials - total_top + 1) / float(total_trials + 2)
    print("%.15s\t\t\t%d\t%d\t%.2f\t%.2f      %d\t%.2f\n" % ("TOTALS", total_trials, total_top, total_top_pct,
                                          total_prob_success, total_consequence, total_consequence_pct))
                                        

# Produces stats and plots for each cont. variables
def cont_analysis(cd):

    # Calculate some simple statistics
    avg_top_prob = ((np.sum(cd.top_events) + 1) / float(len(cd.top_events) + 2))
    avg_col_prob = ((np.sum(cd.consequences) + 1) / float(len(cd.consequences) + 2))
    avg_col_prob_after_top = ((np.sum(cd.consequences) + 1) / float(np.sum(cd.top_events) + 2))
    print("Average TOP Probability: %f" % avg_top_prob)
    print("Average Consequence Probability: %f" % avg_col_prob)
    print("Average Consequence Probability given TOP occurred: %f" % avg_col_prob_after_top)
    print("\n\n")

    for key, values in cd.list_independent_vars_cont.items():

        values_np = np.array(values)
        top_np = np.array(cd.top_events)
        cons_np = np.array(cd.consequences)

        # # Fit a conditional binomial distribution to the data with max-likelihood for the TOP event
        x_range = np.linspace(np.min(values), np.max(values), 100)
        fit_results = None
        try:
            fit_results, _ = function_fitting.max_likelihood_conditional_fit(values_np, top_np) 
        except ValueError as e:
            print("Exception occurred when fitting sigmoid for TOP event:\n%s" % str(e))
        else:
            y_sig_max_likelihood = function_fitting.bounded_sigmoid(x_range, *fit_results.x)
            adjusted_likelihood = y_sig_max_likelihood/avg_top_prob
            print("TOP Sigmoid Fit: ", fit_results)
            print("\n\n")


        #todo add this back for plots

        # Plot conditional probability P(B | D) and adjusted probability P(B | D) / P(B) for TOP barrier
        fig1 = plt.figure(dpi=300)
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.scatter(values_np, top_np, label="Top Events")
        ax1.scatter(values_np, cons_np, label="Collision Events")
        # ax1.scatter(bins_avg["x"], bins_avg["y"]["top"], label="Top Event Prob.")
        ax1.scatter(values_np, top_np, label="Top Events")
        ax1.scatter(np.average(values_np), avg_top_prob, label="Average")
        if fit_results is not None:
            ax1.plot(x_range, y_sig_max_likelihood, label="Sigmoid MLE fit")
            ax1.plot(x_range, adjusted_likelihood, label="Adjusted fit")
            ax1.set_ylim([0.0, np.max(adjusted_likelihood)])
        else:
            ax1.set_ylim([0.0, 1.0])
        ax1.set_xlabel("Indep. Var Percentage")
        ax1.set_ylabel("TOP Event Probability")
        ax1.legend(loc="best")


        # Fit a conditional binomial distribution to the data with max-likelihood for the TOP event
        fit_results = None
        try:
            fit_results, _ = function_fitting.max_likelihood_conditional_fit(values_np, cons_np)
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
        ax2.scatter(values_np, cons_np, label="Collision Events")
        # ax2.scatter(bins_avg["x"], bins_avg["y"]["col"], label="Collision Event Prob.")
        ax2.scatter(np.average(values_np), avg_col_prob, label="Average")
        if fit_results is not None:
            ax2.plot(x_range, collision_y_sigmoid, label="Sigmoid MLE fit")
            ax2.plot(x_range, collision_adj_likelihood, label="Adjusted fit")
            ax2.set_ylim([0.0, np.max(collision_adj_likelihood)])
        else:
            ax2.set_ylim([0.0, 1.0])
        ax2.set_xlabel("Indep. Var Percentage")
        ax2.set_ylabel("Collision Probability")
        ax2.legend(loc="best")

        # Show plots
        plt.show()


def calculate_stats(datafiles):
    """Plot the outcome of each scenario against the average assurance monitor value"""

    cd = CollatedData()
    cd.build(datafiles)
    # cd.show()

    # Datasets with no top events or collisions can cause issues with max likelihood estimation. Print warning.
    if np.sum(cd.top_events) == 0:
        print("WARNING: No Top events occurred in the provided datasets.")
    if np.sum(cd.consequences) == 0:
        print("WARNING: No Consequence events occurred in the provided datasets.")

    if(cd.list_independent_vars_disc):
        disc_anaylsis(cd)
   
    if(cd.list_independent_vars_cont):
        cont_analysis(cd)
    
   
def read_datafile(file_path):
    df = Datafile()
    df.read(file_path)
    return df


if __name__ == "__main__":
    DATA_DIR = "estimation_data"

    description = "calculate stats for resonate\n"

    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument("path", help="specify the location of the data to run analysis on", type=str)
    parser.add_argument("--datafile", "-df", default="uuv", help="select the datafile to use")
    
    arguments = parser.parse_args()

    if(arguments.datafile.lower() == "uuv"):
        data_file_names = glob.glob(os.path.join(arguments.path, "*.bag"))
        Datafile = UUV_Datafile
    elif(arguments.datafile.lower() == "carla"):
        data_file_names = glob.glob(os.path.join(arguments.path, "*.csv"))
        Datafile = CARLA_Datafile
    else:
        raise IOError("unrecognized datafile type")
    
    # TODO: Multithreading did not produce large speedup like expected. Why not? No obvious HW bottleneck.
    # Read Datafiles. Use multiple threads, one thread for each bag file.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Start the read operations and mark each future with its filename
        future_to_file = {executor.submit(read_datafile, file): file for file in data_file_names}

        # As each thread completes, store the resulting datafile and report progress
        completion_count = 0
        datafiles = []
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]

            # try:
            df = future.result()
            # except Exception as exc:
            #     print('%r generated an exception: %s' % (file, exc))
            # else:
            if df is not None:
                datafiles.append(df)

            # Print update every 10 bag files
            completion_count += 1
            if completion_count % 10 == 0:
                print("Processed %d/%d bag files..." % (completion_count, len(future_to_file)))
    print("Finished processing bag files.")

    print("\n")

    calculate_stats(datafiles)
