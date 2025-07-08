# model/sampler.py
import numpy as np

class Sampler:
    def __init__(self, data, output_info):
        self.data = data
        self.model = []
        self.n = len(data)

        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]

        idx = []
        for c, o in zip(col, opt):
            candidates = self.model[c][o]
            if len(candidates) == 0:
                idx.append(np.random.choice(np.arange(self.n)))
            else:
                idx.append(np.random.choice(candidates))

        return self.data[idx]
