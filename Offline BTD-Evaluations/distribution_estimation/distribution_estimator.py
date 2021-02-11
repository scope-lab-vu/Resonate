import pandas as pd
import math

DATA_FILE = "example_data/braking_data.csv"

df = pd.read_csv(DATA_FILE, sep=',')

run1_result_index = 0
run2_result_index = 1
precip_index = 2
num_categories = 5
precip_results = []

# Want to divide the braking dataset into categories bounded by the level of precipitation
# Then count number of trails & failures for each category and use this to estimate probability
step_size = math.ceil(100/num_categories)
steps = int(100/step_size)

print("Lower Bound  |  Upper Bound  |  Num Trials  |  Estimated Probability")
print("-----------------------------------------------------")
for i in range(steps):
    lower_bound = i * step_size
    upper_bound = (i + 1) * step_size
    run_count = 0
    failure_count = 0

    for data_row in df.values:
        if lower_bound <= data_row[precip_index] < upper_bound:
            run_count += 2
            failure_count += data_row[run1_result_index]
            failure_count += data_row[run2_result_index]

    if run_count > 0:
        # Estimate probability using Laplace rule of succession
        est_prob = (run_count - failure_count + 1) / (run_count + 2)
        print("%d\t\t\t%d\t\t\t%d\t\t\t%f" % (lower_bound, upper_bound, run_count, est_prob))
    else:
        print("%d\t\t\t%d\t\t\t%d\t\t\tN/A" % (lower_bound, upper_bound, run_count))
