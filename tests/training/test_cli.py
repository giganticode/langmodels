import unittest
from unittest import mock

from langmodels.lmconfig.defaults import lm_training_config, Gpu
from langmodels.training.__main__ import run


@mock.patch('langmodels.training.cli.train')
class MainTest(unittest.TestCase):
    def test_defaults(self, train_mock):
        argv = ['train', '--config=defaults']
        run(argv)
        train_mock.assert_called_with(tune=False, comet=True,
                                      gpu=Gpu(fallback_to_cpu=False, non_default_device_to_use=0),
                                      lm_training_config=lm_training_config)

    def test_device_comet_cpu(self, train_mock):
        argv = ['train', '--config=defaults', '--fallback-to-cpu', '--tune', '--disable-comet', '--device=3']
        run(argv)
        train_mock.assert_called_with(tune=True, comet=False,
                                      gpu=Gpu(fallback_to_cpu=True, non_default_device_to_use=3),
                                      lm_training_config=lm_training_config)

    def test_short_options(self, train_mock):
        argv = ['train', '--config=defaults', '-pxt', '-d 3']
        run(argv)
        train_mock.assert_called_with(tune=True, comet=False,
                                      gpu=Gpu(fallback_to_cpu=True, non_default_device_to_use=3),
                                      lm_training_config=lm_training_config)


if __name__ == '__main__':
    unittest.main()
