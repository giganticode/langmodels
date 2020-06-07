from langmodels.cli.impl import handle_train
from langmodels import app_name

import docopt_subcommands as dsc

from langmodels import __version__


@dsc.command()
def train_handler(args):
    """usage: {program} train [--config <config>] [--patch <patch>] [--fallback-to-cpu] [--tune] [--disable-comet]
    [--save-every-epoch] [--allow-unks] [--device=<device>]

    Trains a language model according to the given config.

    Options:
      -C, --fallback-to-cpu                        Fallback to cpu if gpu with CUDA-support is not available
      -x, --disable-comet                          Do not log experiment to comet.ml
      -e, --save-every-epoch                       Save the model to the disk after every epoch
      -u, --allow_unks                             Allow unknown tokens
      -t, --tune                                   Training will be done only on a few batches
                                                    (can be used for model params such as batch size to make sure
                                                    the model fits into memory)
      -d <device>, --device=<device>               Device id to use
      -c, --config=<config>                        Path to the json with config to be used to train the model
      -p, --patch=<patch>                          'Patch' to apply to the default lm training config e.g

    """
    handle_train(args)


def run(args):
    dsc.main(app_name, f'{app_name} {__version__}', argv=args, exit_at_end=False)
