from fault_modes import FaultModes
from bowtie_diagram import BowTie


def calc_risk(state):
    # Calculate probabilities from bowtie
    # This will only work for this particular BTD. Need a generalized approach for a long-term solution
    # bowtie = BowTie()
    # p_t1_top = bowtie.prob_t1(state) * (1 - bowtie.prob_b1(state))
    # p_t2_top = bowtie.prob_t2(state) * (1 - bowtie.prob_b2(state))
    # p_top = 1 - ((1 - p_t1_top) * (1 - p_t2_top))
    # p_c1 = p_top * (1 - bowtie.prob_b3(state))
    # return p_c1
    bowtie = bowtie_diagram.BowTie()
    r_t1_top = bowtie.rate_t1(state) * (1 - bowtie.prob_b1(state)) # threat1_rate * (1-p(b1|s))
    r_t2_top = bowtie.rate_t2(state) * (1 - bowtie.prob_b2(state)) # threat2_rate * (1-p(b2|s))
    r_top = r_t1_top + r_t2_top  #add two threats as they are independent
    r_c1 = r_top * (1 - bowtie.prob_b3(state)) # top_threat * (1-p(b3|s))
    return r_c1
