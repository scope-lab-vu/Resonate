from . import bowtie_diagram, fault_modes


def calc_risk(state):
    # Calculate probabilities from bowtie
    # This will only work for this particular BTD. Need a generalized approach for a long-term solution
    bowtie = bowtie_diagram.BowTie()
    r_t1_top = bowtie.rate_t1(state) * (1 - bowtie.prob_b1(state))
    r_t2_top = bowtie.rate_t2(state) * (1 - bowtie.prob_b2(state))
    r_top = r_t1_top + r_t2_top
    r_c1 = r_top * (1 - bowtie.prob_b3(state))
    return r_c1
