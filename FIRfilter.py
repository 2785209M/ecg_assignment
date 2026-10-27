import collections

class FIRfilter:
    def __init__(self, _coefficients):

        self.coefficients = _coefficients
        self.num_taps = len(self.coefficients)
        self.delay_line = collections.deque(maxlen=self.num_taps)

        return

    def dofilter(self, v):

        self.delay_line.appendleft(v)
        result = 0

        for i in range(self.num_taps):
            if i < len(self.delay_line):
                result += self.delay_line[i] * self.coefficients[i]
            else:
                result += 0

        return result

    def dofilteradaptive(self, signal, desired, mu):
            """
            Adaptive filtering using the LMS algorithm.
            :param signal: The input signal !to be filtered out!
            :param noise: The noise signal to be used for adaptation AKA ecg!
            :param mu: The step size for adaptation.
            :return: The filtered signal.
            """
            result = self.dofilter(signal)

            # Update coefficients using LMS algorithm
            error = desired - result
            for i in range(self.num_taps):
                if i < len(self.delay_line):
                    self.coefficients[i] += mu * error * self.delay_line[i]
    
            return error