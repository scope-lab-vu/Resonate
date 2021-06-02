
class Datafile(object):
    def __init__(self):
        self.top_event = None  # Boolean
        self.consequence = None # Boolean
        self.independent_vars_cont = {} # Dictionary, string -> float
        self.independent_vars_disc = {} # Dictionary, string -> float

    def show(self):
        print("Top event Occured:", self.top_event)
        print("Consequence Occurred:", self.consequence)
        for key, value in self.independent_vars_cont.items():
            print("{}: {}".format(key, value))
        for key, value in self.independent_vars_disc.items():
            print("{}: {}".format(key, value))

    def read(self, filepath):
        pass
