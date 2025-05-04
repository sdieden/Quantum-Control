class Pulse:
    liste_pulse = []
    amplitudes_list = set()
    def __init__(self,amplitude,duration):
        self.amplitude = amplitude #amplitude of the pulse
        self.duration = duration #duration of the pulse
        self.stark = None  # if false, evolution must be done on atomic levels, else it must be on Stark level
        if self.amplitude == 0:
            self.stark = False
        else:
            self.stark = True
        self.liste_pulse.append(self)
        self.initial_state = []
        self.final_state = []
        self.amplitudes_list.add(self.amplitude)

    def index_of_amplitude(self):
        return self.amplitudes().index(self.amplitude)

    def amplitudes(self):
        return sorted(tuple(self.amplitudes_list))
