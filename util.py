class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, val, n=1):
        self.count += n
        self.sum += val * n

    def float(self):
        return self.sum / self.count

    def __repr__(self):
        return '%.3f' % (self.sum / self.count)
