import os.path as osp
import shutil
import tempfile

import pytest
import torch
from torch import nn

from torchpack import load_checkpoint, save_checkpoint


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
        x = self.fc(x.view(-1))
        return x

    def verify_params(self, state_dict):
        from collections import OrderedDict
        assert isinstance(state_dict, OrderedDict)
        assert list(state_dict.keys()) == [
            'conv.weight', 'conv.bias', 'fc.weight', 'fc.bias'
        ]


def test_save_checkpoint():
    tmp_dir = tempfile.mkdtemp()
    model = Model()
    epoch = 1
    num_iters = 100
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    save_checkpoint(model, epoch, num_iters, tmp_dir)
    assert osp.isfile(tmp_dir + '/epoch_1.pth')
    chkp = torch.load(tmp_dir + '/epoch_1.pth')
    assert isinstance(chkp, dict)
    assert chkp['epoch'] == epoch
    assert chkp['num_iters'] == num_iters
    model.verify_params(chkp['state_dict'])
    save_checkpoint(
        model,
        epoch,
        num_iters,
        tmp_dir,
        filename_tmpl='test_{}.pth',
        optimizer=optimizer)
    assert osp.isfile(tmp_dir + '/test_1.pth')
    chkp = torch.load(tmp_dir + '/test_1.pth')
    assert isinstance(chkp, dict)
    assert chkp['epoch'] == epoch
    assert chkp['num_iters'] == num_iters
    model.verify_params(chkp['state_dict'])
    shutil.rmtree(tmp_dir)


def test_load_checkpoint():
    model = Model()
    with pytest.raises(IOError):
        load_checkpoint(model, 'non_exist_file')
