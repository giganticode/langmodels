from pytest import fixture
from pytest_mock.plugin import MockFixture

import langmodels.cli.spec as cli_spec
import langmodels.cli.impl as cli_impl
from langmodels.lmconfig.datamodel import DeviceOptions, LMTrainingConfig


@fixture
def train_func_mocker(mocker: MockFixture):
    mocker.patch('langmodels.cli.impl.train')
    return mocker


def test_defaults(train_func_mocker):
    argv = ['train']
    cli_spec.run(argv)
    cli_impl.train.assert_called_with(allow_unks=False, tune=False, comet=True, save_every_epoch=False,
                                 device_options=DeviceOptions(fallback_to_cpu=False, non_default_device_to_use=0),
                                 training_config=LMTrainingConfig())


def test_device_comet_cpu(train_func_mocker):
    argv = ['train', '--fallback-to-cpu', '--tune', '--disable-comet', '--device=3', '--save-every-epoch']
    cli_spec.run(argv)
    cli_impl.train.assert_called_with(allow_unks=False, tune=True, comet=False, save_every_epoch=True,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig())


def test_short_options(train_func_mocker):
    argv = ['train', '-Cxte', '-d 3']
    cli_spec.run(argv)
    cli_impl.train.assert_called_with(allow_unks=False, tune=True, comet=False, save_every_epoch=True,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig())
