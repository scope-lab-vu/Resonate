def read_bag():
    data = {"top_event": Boolean,
            "consequence": Boolean,
            "independent_vars_cont": dict String->Float,
            "independent_vars_disc": dict String->Int/Boolean}

    data = {"top_event": True,
            "consequence": False,
            "independent_vars_cont": {"thruster_degredation": x1, "water_current": x2, ...},
            "independent_vars_disc": {"fault mode 1": False, "fault mode 2": True, ...}}


def analyze_datafiles():
    # Collate data from all bag files
    collated_data = {"top_event": list Boolean,
                    "consequence": list Boolean,
                    "independent_vars_cont": list dict String->Float,
                    "independent_vars_disc": list dict String->Int/Boolean}
    for data in datafiles:
        collated_data["top_event"].append(data["top_event"])
        collated_data["top_event"].append(data["top_event"])

    # Max Likelihood Fit for continuous variables
    mle_fits = []
    for var_name, var_value in iteritems(collated_data["independent_vars_cont"]):
        fit = max_likelihood_fit(var)
        mle_fits.append(fit)

        # Plot fit results
        plot(fit)
        plot.title("Varible X fit")

    # Laplace rule of succession for discrete variables
    probs = []
    for var_name, var_value in iteritems(collated_data["independent_vars_disc"]):
        prob = ((np.sum(var_value) + 1) / float(len(var_value) + 2))
        probs.append(probs)

        print("VAR_NAME, Probability" % var_name, prob)

