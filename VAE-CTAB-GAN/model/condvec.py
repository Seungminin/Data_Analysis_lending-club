# model/condvec.py
import numpy as np

class Condvec:
    def __init__(self, data, output_info):
        self.model = []
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        self.p_log_sampling = []

        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)
                log_freq = np.log(freq + 1)
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch_size):
        if self.n_col == 0:
            return None, None, None, None

        vec = np.zeros((batch_size, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch_size)
        mask = np.zeros((batch_size, self.n_col), dtype='float32')
        mask[np.arange(batch_size), idx] = 1

        opt1prime = []
        for i in idx:
            p = self.p_log_sampling[i] + 1e-6
            p = p / p.sum()
            opt1prime.append(np.random.choice(np.arange(len(p)), p=p))

        opt1prime = np.array(opt1prime)
        for i in range(batch_size):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime
