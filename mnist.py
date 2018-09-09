from torchvision import datasets
import torchbaker as tb
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x, y = x
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), y


class MNISTRecipe(tb.Recipe):
    def __init__(self):
        self.max_iter_num = 10 ** 4
        self.batch_size = 32
        self.max_checkpoint_num = 5
        self.checkpoint_dir = './checkpoints/mnist'
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,
                                        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True,
                                       transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
        self.dataloaders['train'] = DataLoader(mnist_trainset, self.batch_size, True, num_workers=4)
        self.dataloaders['test'] = DataLoader(mnist_testset, self.batch_size, False, num_workers=4)
        self.modules['m'] = model = Net()
        self.modules['nll'] = F.nll_loss
        self.optimizers['opt'] = optim.Adam(model.parameters(), lr=1e-3)

        # custom passes
        self.train_passes = [
            tb.Pass('[in] => m => [t1,t2] => [t3] => [o,t] => nll => [loss]', lambda: self.variables.loss,
                    self.optimizers)]
        self.test_passes = ['[in] => m => [o,t]']

        # callbacks
        def get_correct_num(output, target):
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            return pred.eq(target.view_as(pred)).sum().item()

        @self.register_callback('train_iter_finish')
        @self.do_every_iter(200)
        def show():
            correct = get_correct_num(self.variables.o, self.variables.t)
            acc = correct / self.batch_size
            print(
                'iter: {}/{}, Acc: {}/{} = {}'.format(self.iter_num, self.max_iter_num, correct, self.batch_size, acc))

        @self.register_callback('test_begin')
        def test_begin():
            self.test_correct = 0

        @self.register_callback('test_iter_finish')
        def test_iter_finish():
            self.test_correct += get_correct_num(self.variables.o, self.variables.t)

        @self.register_callback('test_finish')
        def test_finish():
            c, n = self.test_correct, len(self.dataloaders['test'].dataset)
            print('Test Acc: {}/{} = {}'.format(c, n, c / n))

        @self.register_callback('train_iter_finish')
        @self.do_every_iter(200)
        @self.register_callback('train_finish')
        def do_save():
            self.save()

    def load(self):
        try:
            super(MNISTRecipe, self).load()
        except tb.exceptions.CheckpointError:
            print('No checkpoint found, random initialize')


if __name__ == '__main__':
    b = tb.Baker()
    b.prepare_recipe(MNISTRecipe())
    b.train()
    b.test()
