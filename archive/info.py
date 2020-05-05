# from __future__ import print
import numpy as np

class Packet(object):
    def __init__(self, phase_type, x1, y1, slope, run):
        self.phase_type = phase_type
        self.x1 = x1
        self.y1 = y1
        self.slope = slope
        self.run = run

    def dump(self):
        print('Info. packet --> phase type: {}, (x1,y1): ({},{}), slope: {}, run: {}'.format(
                self.phase_type, self.x1, self.y1, self.slope, self.run))

class Information(object):

    def linesInfo(self, lines):
        """Function to measure the information content of hough lines.
        
        Arguments:
            lines {[HoughLine]} -- [hough lines]
        """
        # TODO: Compute weight info bits by using some quantizations methods
        # NOTE: Currently, we assume that each weight will be quantized to 16 bits
        import math
        assert lines

        def next_power_of_2(x):
            return 1 if x == 0 else 2**math.ceil(math.log(x, 2))

        x1, y1, x2, y2, w = [], [], [], [], []
        for i in lines:
            x1.append(i.x1)
            y1.append(i.y1)
            x2.append(i.x2)
            y2.append(i.y2)
            w.append(i.weight)
        
        bits_x1 = math.log(next_power_of_2(np.max(x1)-np.min(x1)), 2) * (len(x1)+1)
        bits_y1 = math.log(next_power_of_2(np.max(y1)-np.min(y1)), 2) * (len(y1)+1)
        bits_x2 = math.log(next_power_of_2(np.max(x2)-np.min(x2)), 2) * (len(x2)+1)
        bits_y2 = math.log(next_power_of_2(np.max(y2)-np.min(y2)), 2) * (len(y2)+1)
        # len + 1 to save the initial offset. 
        return bits_x1 + bits_y1 + bits_x2 + bits_y2 + (len(w)*16)

    def encode(self, lines):
        """Encodes the hough line data to fit in the necessary amount of bits.
        
        Arguments:
            lines {HoughLine} -- hough lines 
        
        Raises:
            NotImplementedError -- :|
        """

        raise NotImplementedError

    def countBits(self, Packet):
        #TODO: Should it be string instead? that is simply a len(string) op?
        """ Counts the number of bits per packet, or per string of information.
        """
        raise NotImplementedError

    def getString(self, packet_list):
        """ Compute a string with a list of Packets
        """
        raise NotImplementedError
