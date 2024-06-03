import matplotlib.pyplot as plt
import os

class PltHelper:
    def __init__(self):
        self.epoches = []
        self.train_loss = []
        self.validation_loss = []
        pass

    def append(self, epoch, train_loss, validation_loss):
        self.epoches.append(epoch)
        self.train_loss.append(train_loss)
        self.validation_loss.append(validation_loss)

        plt.clf()

        plt.plot(self.epoches, self.train_loss, color='r', label='train_loss')
        plt.plot(self.epoches, self.validation_loss, color='b', label='validation_loss')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")

        path = "./loss_record/"

        if not os.path.exists(path):
            os.makedirs(path)

        name = "epoch_" + str(epoch) + ".png"
        plt.savefig(path + name)


if __name__ == "__main__":
    m = PltHelper()
    for epoch in range(10):
        m.append(epoch, 0.01 * epoch, 0.02 * epoch)