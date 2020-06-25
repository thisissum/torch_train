from collections import deque


class Context(object):
    """
    Context class that contains infomation in training process
    """
    def __init__(self, record_length=100):
        self.context_info = {'train_loss': deque(maxlen=record_length),
                             'validation_loss': deque(maxlen=record_length)}


    def __getitem__(self, item):
        if item in ('train_loss', 'validation_loss'):
            return np.mean(self.context_info[item])
        else:
            return self.context_info[item]


    def update(self, info):
        for key, val in info.items():
            if key in ('train_loss', 'validation_loss'):
                self.context_info[key].append(val)
            else:
                self.context_info[key] = val


    def get_fields(self):
        return self.context_info.keys()
