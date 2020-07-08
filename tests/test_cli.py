from pathlib import Path
from unittest.mock import Mock

from pytest import fixture
from pytest_mock.plugin import MockFixture

import langmodels.cli.spec as cli_spec
import langmodels.cli.impl as cli_impl
from langmodels.lmconfig.datamodel import LMTrainingConfig
from langmodels.util.cuda import DeviceOptions


@fixture
def train_func_mocker(mocker: MockFixture):
    mocker.patch('langmodels.cli.impl.train')
    return mocker


@fixture
def evaluate_func_mocker(mocker: MockFixture):
    mocker.patch('langmodels.cli.impl.evaluate_on_path')
    mocker.patch('langmodels.cli.impl.load_from_path')
    return mocker


def test_defaults(train_func_mocker):
    argv = ['train']
    cli_spec.run(argv)
    cli_impl.train.assert_called_with(allow_unks=False, tune=False, comet=True, save_every_epoch=False,
                                 device_options=DeviceOptions(fallback_to_cpu=False, non_default_device_to_use=0),
                                 training_config=LMTrainingConfig(), output_path=None, rewrite_output=False)


def test_device_comet_cpu(train_func_mocker):
    argv = ['train', '--fallback-to-cpu', '--tune', '--disable-comet',
            '--device=3', '--save-every-epoch', '--output-path=/home/username', '--rewrite-output']
    cli_spec.run(argv)
    cli_impl.train.assert_called_with(allow_unks=False, tune=True, comet=False, save_every_epoch=True,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig(), output_path='/home/username', rewrite_output=True)


def test_short_options(train_func_mocker):
    argv = ['train', '-Cxtef', '-d 3']
    cli_spec.run(argv)
    cli_impl.train.assert_called_with(allow_unks=False, tune=True, comet=False, save_every_epoch=True,
                                 device_options=DeviceOptions(fallback_to_cpu=True, non_default_device_to_use=3),
                                 training_config=LMTrainingConfig(), output_path=None, rewrite_output=True)



def test_evaluate_with_defaults(evaluate_func_mocker):
    mocked_model = Mock()
    cli_impl.load_from_path.return_value = mocked_model

    argv=['evaluate', '/path/to/model', '--path', '/path/to/evaluate', '--output-path', '/path/to/output']
    cli_spec.run(argv)
    cli_impl.load_from_path.assert_called_with('/path/to/model', after_epoch=None,  device=0)
    cli_impl.evaluate_on_path.assert_called_with(mocked_model, '/path/to/evaluate', Path('/path/to/output'), full_tokens=True)


def test_evaluate_with_all_options(evaluate_func_mocker):
    mocked_model = Mock()
    cli_impl.load_from_path.return_value = mocked_model

    argv=['evaluate', '/path/to/model', '--after-epoch', '43', '--path', '/path/to/evaluate', '--output-path', '/path/to/output', '--sub-tokens', '--batch-size', '13', '--device' ,'1']
    cli_spec.run(argv)
    cli_impl.load_from_path.assert_called_with('/path/to/model', after_epoch='43', device=1)
    cli_impl.evaluate_on_path.assert_called_with(mocked_model, '/path/to/evaluate', Path('/path/to/output'), full_tokens=False, batch_size=13)


def test_evaluate_with_short_options(evaluate_func_mocker):
    mocked_model = Mock()
    cli_impl.load_from_path.return_value = mocked_model

    argv=['evaluate', '/path/to/model', '-e', '43', '-p', '/path/to/evaluate', '-o', '/path/to/output', '-sb', '13', '-d', '1']
    cli_spec.run(argv)
    cli_impl.load_from_path.assert_called_with('/path/to/model', after_epoch='43', device=1)
    cli_impl.evaluate_on_path.assert_called_with(mocked_model, '/path/to/evaluate', Path('/path/to/output'), full_tokens=False, batch_size=13)