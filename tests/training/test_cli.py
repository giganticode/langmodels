from pytest import fixture
from pytest_mock.plugin import MockFixture

import langmodels.training.cli as cli
from langmodels.lmconfig.datamodel import LMTrainingConfig
from langmodels.cuda_util import DeviceOptions


@fixture
def train_func_mocker(mocker: MockFixture):
    mocker.patch('langmodels.training.cli.train')
    return mocker


def test_defaults(train_func_mocker):
    argv = ['train']
    cli.run(argv)
    cli.train.assert_called_with(allow_unks=False, tune=False, comet=True, save_every_epoch=False,
                                 device_options=DeviceOptions(fallback_to_cpu=False, non_default_device_to_use=0),
                                 training_config=LMTrainingConfig(), output_path=None, rewrite_output=False)


def test_device_comet_cpu(train_func_mocker):
    argv = ['train', '--fallback-to-cpu', '--tune', '--disable-comet',
            '--device=3', '--save-every-epoch', '--output-path=/home/username', '--rewrite-output']
    cli.run(argv)
    cli.train.assert_called_with(allow_unks=False, tune=True, comet=False, save_every_epoch=True,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig(), output_path='/home/username', rewrite_output=True)


def test_short_options(train_func_mocker):
    argv = ['train', '-Cxtef', '-d 3']
    cli.run(argv)
    cli.train.assert_called_with(allow_unks=False, tune=True, comet=False, save_every_epoch=True,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig(), output_path=None, rewrite_output=True)
