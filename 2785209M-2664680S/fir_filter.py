import collections

class FIRfilter:
    def __init__(self, _coefficients):
        # Store the coefficients in this filter object (denoted by "self.")
        self.coefficients = _coefficients
        # Store the number of taps (== length of coefficients array)
        self.num_taps = len(self.coefficients)
        # Initialize a double-ended queue with a length of num_taps
        self.delay_line = collections.deque(maxlen=self.num_taps)

        return

    def dofilter(self, v):
        # Append incoming samples to the first index of the double-ended queue
        self.delay_line.appendleft(v)
        # initialize an accumulator for this window
        result = 0

        # Perform a convolution sum - multiply each sample in the buffer by its corresponding coefficient,
        # and add the result to the accumulator
        for i in range(len(self.delay_line)):
            result += self.delay_line[i] * self.coefficients[i]

        return result

    def dofilteradaptive(self, signal, noise, learningRate):
            """
            Adaptive filtering using the LMS algorithm.
            - signal: The input signal, to be filtered out
            - desired: The desired signal to be used for adaptation AKA ecg
            - mu: The step size for adaptation
            - result: The filtered signal
            """

            # perform a convolution sum to modify the amplitude and phase of the pure input signal
            filtered_signal = self.dofilter(signal)

            # calculate the error (the clean signal)
            error = noise - filtered_signal
            for i in range(len(self.delay_line)):
                # Update coefficients using LMS algorithm
                self.coefficients[i] += learningRate * error * self.delay_line[i]

            # Return the error signal (the difference between the desired signal and the filtered output)
            return error