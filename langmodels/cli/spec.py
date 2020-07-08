from langmodels.cli.impl import handle_train, handle_evaluation
from langmodels import app_name

import docopt_subcommands as dsc

from langmodels import __version__


@dsc.command()
def train(args):
    """usage: {program} train [--config <config>] [--patch <patch>] [--fallback-to-cpu] [--tune] [--disable-comet]
[--save-every-epoch] [--allow-unks] [--device=<device>] [--output-path=<path>] [--rewrite-output]

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
      -o, --output-path=<path>                     Path to where the models and metrics will be saved.
                                                   If not specified:
                                                   On Mac OS X:
                                                       ~/Library/Application Support/langmodels/<langmodels-version>/modelzoo/<run-id>
                                                   On Unix:
                                                       ~/.local/share/langmodels/<langmodels-version>/modelzoo/<run-id>
                                                       or if XDG_DATA_HOME is defined:
                                                       $XDG_DATA_HOME/langmodels/<langmodels-version>/modelzoo/<run-id>
                                                   <run-id> is generated based on the current timestamp and is normally
                                                   unique unless multiple ids are generated at the same second
                                                   (if multiple experiments are run at the same time)

       -f, --rewrite-output                        Rewrite already existing output
    """
    handle_train(args)


@dsc.command()
def evaluate(args):
    """usage: {program} evaluate <path-to-model> [--after-epoch <after-epoch>] -p <path> -o <path-out> [--sub-tokens]
[--batch-size <batch-size>] [--device=<device>]

    Evaluates the language model on the given corpus

    Options:
      <path-to-model>
      -e, --after-epoch <after-epoch>
      -p, --path <path>
      -o, --output-path <path-out>
      -s, --sub-tokens
      -b, --batch-size <batch-size>
      -d <device>, --device=<device>               Device id to use

    """
    handle_evaluation(args)


def run(args):
    dsc.main(app_name, f'{app_name} {__version__}', argv=args, exit_at_end=False)
