from risk_calculation import calc_risk
import csv
import pprint

DATA_FILE = "simulation_Scene_with_no_rain.csv"

ENV_LABELS = ["precipitation",
              "precipitation_deposits",
              "cloudiness",
              "wind_intensity",
              "sun_azimuth_angle",
              "sun_altitude_angle",
              "fog_density",
              "fog_distance",
              "wetness"]
FAULT_LABEL = "fault_type"
MONITOR_LABELS = []

EXAMPLE_MONITOR_VALUES = range(-5, 40)


# Read data from CSV file
with open(DATA_FILE) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    labels = None
    data_rows = []
    for i, row in enumerate(reader):
        if i == 0:
            labels = row
        else:
            data_rows.append(row)

# Parse CSV data into state structure
states = []
for data_row in data_rows:
    state = {"environment": {}, "fault_modes": None, "monitor_values": {}}
    for i, data_point in enumerate(data_row):
        label = labels[i]
        if label in ENV_LABELS:
            state["environment"][label] = data_point
        elif label == FAULT_LABEL:
            state["fault_modes"] = int(data_point)
        elif label in MONITOR_LABELS:
            state["monitor_values"][label] = data_point
    states.append(state)

pp = pprint.PrettyPrinter()
for state in states:
    print("State: ")
    pp.pprint(state)

    # For each fixed state measurement, try out several example monitor values
    print("Calculated probability of consequence for various AM values: ")
    state["monitor_values"] = {"lec_martingale": None}
    for value in EXAMPLE_MONITOR_VALUES:
        state["monitor_values"]["lec_martingale"] = value
        print(calc_risk(state))

    print("\n\n")
