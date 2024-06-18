import tempfile
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchpack import Runner


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(5)
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def batch_processor(model, data, train_mode):
    p_img, = data
    res = model(p_img)
    loss = res.mean()
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=p_img.size(0))
    return outputs


class TestRunner(object):

    @classmethod
    def setup_class(cls):
        cls.model = Model()
        cls.train_dataset = TensorDataset(torch.rand(10, 2, 5, 5))
        cls.val_dataset = TensorDataset(torch.rand(3, 2, 5, 5))

    def test_init(self):
        optimizer = dict(
            algorithm='SGD',
            args=dict(lr=0.001, momentum=0.9, weight_decay=5e-4))
        work_dir = tempfile.mkdtemp()
        runner = Runner(self.model, optimizer, batch_processor, work_dir)
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=5, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=3, shuffle=False)
        runner.run(
            [train_loader, val_loader], [('train', 1), ('val', 1)],
            max_epoch=2)
