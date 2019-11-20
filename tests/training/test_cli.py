from pytest import fixture
from pytest_mock.plugin import MockFixture

import langmodels.training.cli as cli
from langmodels.lmconfig.datamodel import DeviceOptions, LMTrainingConfig
from langmodels.training.__main__ import run


@fixture
def train_func_mocker(mocker: MockFixture):
    mocker.patch('langmodels.training.cli.train')
    return mocker


def test_defaults(train_func_mocker):
    argv = ['train']
    run(argv)
    cli.train.assert_called_with(tune=False, comet=True,
                                 device_options=DeviceOptions(fallback_to_cpu=False, non_default_device_to_use=0),
                                 training_config=LMTrainingConfig())


def test_device_comet_cpu(train_func_mocker):
    argv = ['train', '--fallback-to-cpu', '--tune', '--disable-comet', '--device=3']
    run(argv)
    cli.train.assert_called_with(tune=True, comet=False,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig())


def test_short_options(train_func_mocker):
    argv = ['train', '-Cxt', '-d 3']
    run(argv)
    cli.train.assert_called_with(tune=True, comet=False,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig())
