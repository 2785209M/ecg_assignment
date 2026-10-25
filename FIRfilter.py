import collections

class FIRfilter:
    def __init__(self, _coefficients):

        self.coefficients = _coefficients
        self.delay_line = collections.deque(maxlen=len(self.coefficients))

        return

    def dofilter(self, v):

        self.delay_line.appendleft(v)
        result = 0

        for i in range(len(self.coefficients)):
            if i < len(self.delay_line):
                result += self.delay_line[i] * self.coefficients[i]
            else:
                result += 0

        return result