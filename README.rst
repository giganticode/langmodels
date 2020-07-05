======================
Giganticode-langmodels
======================

|CI|

This is a repository for *neural language models (LMs)* trained on a large corpus of source code
and a toolkit to work with such models.

Features:

- Autocompletion and bug prediction with pre-trained models we provide;
- Use the pre-trained models as a starting point for transfer learning or further fine-tuning;
- Training a model from scratch by choosing one of many available corpus pre-processing and training options.

This project uses fastai_ and pytorch_ libraries for NN training/inference.
For corpus preprocessing `giganticode-dataprep <https://github.com/giganticode/dataprep>`_ is used.

.. _fastai: https://www.fast.ai
.. _pytorch: https://pytorch.org

.. contents:: **Contents**
  :backlinks: none

Quick start
===========

Prerequisites
-------------

* Python version >= 3.6 required!

Installation
------------

pip (PyPI)
~~~~~~~~~~

|PyPI|

.. code-block:: bash

   pip install giganticode-langmodels

Build from source
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com//giganticode/langmodels
    cd langmodels
    python -m venv langmodels-venv
    source langmodels-venv/bin/activate
    pip install -r requirements.txt


Note for windows users:
~~~~~~~~~~~~~~~~~~~~~~~

The library is no longer tested under Windows but most of the functionality is expected to work.

Loading pre-trained models
==========================
Loading a default pre-trained model
-----------------------------------

.. code-block:: python

    >>> import langmodels.repository as repo
    >>> trained_model = repo.load_default_model()
    20...

Other model loading options
---------------------------

**To see which models are available, you can call `list_pretrained_models` function.**

Set ``cached`` parameter to ``True`` (default is ``False``) to display only cached LMs (e.g. if offline).

.. code-block:: python

    >>> import langmodels.repository as repo
    >>> repo.list_pretrained_models(cached=False)
    <BLANKLINE>
      ID                                        BPE_MERGES  LAYERS_CONFIG         ARCH      BIN_ENTROPY        TRAINING_TIME_MINUTES_PER_EPOCH  N_EPOCHS  BEST_EPOCH  SIZE_ON_DISK_MB  TAGS
    <BLANKLINE>
      langmodel-large-split_10k_2_1024_191007.  10k         1024/2/1024=27726250  AWD_LSTM  2.1455788479       1429                             6         5           350              ['BEST', 'DEFAULT']
      112241_-_langmodel-large-split_10k_2_102
      4_191022.141344_new
      langmodel-large-split_10k_1_512_190926.1  10k         512/1/512=0           AWD_LSTM  2.69019493253      479                              9         8           91               ['MEDIUM']
      20146_new
      langmodel-small-split-reversed_10k_1_512  10k         512/1/512=7180977     GRU       4.249997138977051  2                                100       97          51               ['BEST_SMALL']
      _200117.095729
      langmodel-small-split_10k_1_512_190906.1  10k         512/1/512=0           AWD_LSTM  4.73768141172      4                                19        18          84               ['TINY']
      54943_new
      dev_10k_1_10_190923.132328_new            10k         10/1/10=7172          AWD_LSTM  9.15688191092      0                                0         -1          1                ['RANDOM']
    <BLANKLINE>

Use ``query_all_models`` method to get a list of ``ModelDescription`` objects

.. code-block:: python

    >>> import langmodels.repository as repo
    >>> repo.query_all_models()[0]
    ModelSummary(id='langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344_new', bpe_merges='10k', layers_config='1024/2/1024=27726250', arch='AWD_LSTM', bin_entropy=2.1455788479, training_time_minutes_per_epoch=1429, n_epochs=6, best_epoch=5, size_on_disk_mb=350, tags=['BEST', 'DEFAULT'])


**A model can be loaded by tag or by id.**

You can specify if you want to load a model to CPU despite having cuda-supported GPU with `force_use_cpu` parameter
(defaults to ``False``). If cuda-supported GPU is not available, this parameter is disregarded.

.. code-block:: python

    >>> trained_model = repo.load_model_with_tag('BEST')
    2...
    >>> trained_model = repo.load_model_by_id('dev_10k_1_10_190923.132328_new', force_use_cpu=True)
    2...


Also, you can use a lower-level API to load a model by path :

.. code-block:: python

    >>> import os
    >>> from langmodels import project_dir
    >>> path_to_model = os.path.join(project_dir, 'data', 'models', 'dev_10k_1_10_190923.132328')

    >>> trained_model = repo.load_from_path(path_to_model)
    2...

Inference
=========
Autocompletion
--------------

Example

.. code-block:: python

    >>> import langmodels.repository as repo
    >>> trained_model = repo.load_default_model()
    2...
    >>> trained_model.feed_text('public static main() { if', extension='java')

    # this does not change the state of the model:
    >>> trained_model.predict_next_full_token(n_suggestions=5)
    [('(', 0.67...), (',', 0.23...), ('{', 0.016...), ('new', 0.01...), ('}', 0.01...)]

    # adding more context:
    >>> trained_model.feed_text('(', extension='java')
    >>> trained_model.predict_next_full_token(n_suggestions=3)
    [('(', 0.15...), ('1', 0.14...), ('setLength', 0.03...)]


    # resetting the state of the model (make it forget the context)
    >>> trained_model.reset()
    >>> trained_model.predict_next_full_token(n_suggestions=5)
    [('new', 0.05...), ('.', 0.04...), ('this', 0.04...), ('*', 0.01...), ('gle', 0.01...)]


Bug prediction based on per-line entropies evaluation
-----------------------------------------------------

An LM can be used to calculate cross-entropies for each line of a file. High values can give an idea about
unusual/suspicious chunks of code [[1]](#1).

Check section [LM Evaluation](#lm-evaluation) section to learn how to calculate
cross-entropy for a project/file/string,

Check our `vsc plugin <https://github.com/giganticode/lm-powered>`_ for highlighting suspicious code.


Model Training
==============

Python API
----------

.. code-block:: python

    >>> import os
    >>> from langmodels import project_dir
    >>> path_to_corpus = os.path.join(project_dir, 'data', 'dev')

    >>> from langmodels.training.training import train
    >>> from langmodels.lmconfig.datamodel import *

    >>> train(LMTrainingConfig(corpus=Corpus(path=path_to_corpus))) # doctest: +SKIP

More parameters to customize corpus pre-processing, NN architecture, and the training process can be specified:

.. code-block:: python

    >>> import os
    >>> from langmodels import project_dir
    >>> path_to_corpus = os.path.join(project_dir, 'data', 'dev')

    >>> from langmodels.training.training import train
    >>> from langmodels.lmconfig.datamodel import *

    >>> train(LMTrainingConfig(corpus=Corpus(path=path_to_corpus), prep_function=PrepFunction(options=PrepFunctionOptions(no_com=False, no_unicode=True)), arch=GruArch(n_layers=2), training=Training(schedule=RafaelsTrainingSchedule(max_epochs=1))))  # doctest: +SKIP

Below you can see all the default parameters specified explicitly:

.. code-block:: python

    >>> import os
    >>> from langmodels import project_dir
    >>> path_to_corpus = os.path.join(project_dir, 'data', 'dev')

    >>> from langmodels.lmconfig.datamodel import *
    >>> from langmodels.training.training import train

    >>> train(LMTrainingConfig(base_model=None, bs=32, corpus=Corpus(path=path_to_corpus, extensions="java"), prep_function=PrepFunction(corpus_api.bpe, ['10k'], PrepFunctionOptions(no_com=False, no_unicode=True, no_spaces=True, max_str_length=sys.maxsize)), arch=LstmArch(bidir=False, qrnn=False, emb_sz=1024, n_hid=1024, n_layers=3,drop=Dropouts(multiplier=0.5, oute=0.02, outi=0.25, outh=0.15, w=0.2, out=0.1),tie_weights=True, out_bias=True),bptt=200,training=Training(optimizer=Adam(betas=(0.9, 0.99)),files_per_epoch=50000,gradient_clip=0.3,activation_regularization=ActivationRegularization(alpha=2., beta=1.),schedule=RafaelsTrainingSchedule(init_lr=1e-4, mult_coeff=0.5, patience=0,max_epochs=1, max_lr_reduction_times=6),weight_decay=1e-6)), device_options=DeviceOptions(fallback_to_cpu=True), comet=False)
    2...
    <langmodels.model.model.TrainedModel object at ...

CLI API
-------

Training can be run from command line as simple as running ``train`` command passing path to the config in json format
as ``--config`` param. To override values in the json file (or default values if ``--config`` param is not specified),
you can use ``--patch`` param.

.. code-block:: shell

    langmodels train --config="/path/to/json/config.json" --patch="bs=64,arch.drop.multiplier=3.0"


If neither ``--config`` nor ``--patch`` params are specified, the training will be running with the default parameters.
The json with the default parameters would look like follows:



Most probably, you would have to override at least the ``corpus.path`` value.

For more options, run:

.. code-block:: shell

    langmodels train --help

Evaluation
==========

Basic usage
-----------

*Langmodels* provides an API to evaluate the performance of a language model on a given string, file, or corpus.

.. code-block:: python

    >>> from langmodels.evaluation import evaluate_on_string, evaluate_on_file, evaluate_on_path
    >>> from pathlib import Path
    >>> import tempfile

    # Resetting model's state to make evaluation reproducible
    >>> trained_model.reset()

    # Evaluate on a string
    >>> evaluate_on_string(trained_model, 'import java.lang.collections;')
    {'n_samples': 7, 'Entropy': 8.4...}

    # Evaluate on a file
    >>> file = Path(project_dir) /'data' /'dev' /'valid' /'StandardDataTypeEmitter.java'
    >>> evaluate_on_file(trained_model, file)
    {'n_samples': 1528, 'Entropy': 15.8...}

    #Evaluate on a coprus
    >>> path = Path(project_dir) /'data' /'dev' /'valid'
    >>> output_path = Path(tempfile.TemporaryDirectory().name)

    >>> evaluate_on_path(trained_model, path, save_to=output_path)
    2...
    {'n_samples': 1647, 'Entropy': 16.0...}


Evaluation on a big corpora can take a lot of time. Therefore, the evaluation result data is saved to the disk.
Path to the evaluation data can be specified by ``save_to`` parameter. It can be loaded as follows:

.. code-block:: python

    >>> from langmodels.evaluation import EvaluationResult

    >>> evaluation = EvaluationResult.from_path(output_path)

For flexibility, one can use ``Pandas DataFrame API`` to manipulate with evaluation result data:
``EvaluationResult`` is simply a wrapper around ``DataFrame`` which can be accesses via  ``data`` property:

.. code-block:: python

    >>> evaluation.data
                                                                     n_samples                                            example     Entropy
    TokenType           SubtokenNumber Project
    ClosingBracket      1              StandardDataTypeEmitter.java        126                                              )</t>   20.6...
    ClosingCurlyBracket 1              StandardDataTypeEmitter.java         22                                              }</t>    6.0...
    Identifier          1              StandardDataTypeEmitter.java        169                                          write</t>    7.6...
                        2              StandardDataTypeEmitter.java        220                                          sin|k</t>   17.6...
                        3              StandardDataTypeEmitter.java         24                           construct|or|Factory</t>   32.4...
                        4              StandardDataTypeEmitter.java         28                        visit|or|Type|Arguments</t>   44.7...
                        5              StandardDataTypeEmitter.java         57                  em|it|Parameter|ized|TypeName</t>   55.9...
                        6              StandardDataTypeEmitter.java          2                   Standard|Data|Type|E|mit|ter</t>   74.8...
                        7              StandardDataTypeEmitter.java          8               em|it|Base|Class|And|Inter|faces</t>   91.1...
    KeyWord             1              StandardDataTypeEmitter.java         69                                            for</t>    7.3...
    MultilineComment    1              Licence.java                         57                                              /</t>    7.7...
                                       StandardDataTypeEmitter.java         87                                              /</t>    7.5...
                        2              Licence.java                         32                                           th|e</t>   21.4...
                                       StandardDataTypeEmitter.java         42                                           ad|t</t>   21.4...
                        3              Licence.java                         19                                  li|mit|ations</t>   33.9...
                                       StandardDataTypeEmitter.java         22                                      em|it|ter</t>   33.4...
                        4              Licence.java                         10                                     L|ic|en|se</t>   42.7...
                                       StandardDataTypeEmitter.java         10                                     L|ic|en|se</t>   42.7...
                        5              StandardDataTypeEmitter.java          1                            Data|Type|E|mit|ter</t>   53.2...
    NonCodeChar         1              StandardDataTypeEmitter.java         55                                              @</t>    2.4...
    One                 1              StandardDataTypeEmitter.java          1                                              1</t>    6.9...
    OpeningBracket      1              StandardDataTypeEmitter.java        126                                              (</t>    7.1...
    OpeningCurlyBracket 1              StandardDataTypeEmitter.java         22                                              {</t>    7.2...
    Operator            1              StandardDataTypeEmitter.java        252                                              .</t>    6.7...
    Semicolon           1              StandardDataTypeEmitter.java        119                                              ;</t>    6.5...
    SpecialToken        1              Licence.java                          1                                          <EOF></t>   10.9...
                                       StandardDataTypeEmitter.java          1                                          <EOF></t>   10.3...
    StringLiteral       1              StandardDataTypeEmitter.java          9                                            "."</t>    8.0...
                        2              StandardDataTypeEmitter.java         11                                        "\n|\n"</t>    7.0...
                        3              StandardDataTypeEmitter.java          7                                       " |{|\n"</t>   21.8...
                        4              StandardDataTypeEmitter.java          3                              " |implement|s| "</t>   28.9...
                        5              StandardDataTypeEmitter.java          9                                   " |{|\|n|\n"</t>   42.8...
                        7              StandardDataTypeEmitter.java          5                      "   |  | |@|Overrid|e|\n"</t>   54.9...
                        8              StandardDataTypeEmitter.java          4                 "|Gener|ating| |data| |type| "</t>   70.1...
                        9              StandardDataTypeEmitter.java          1              "   |  | |Result|Type| |_|case|("</t>   73.2...
                        10             StandardDataTypeEmitter.java          3                   "   |  | |v|o|id| |_|case|("</t>   81.2...
                        11             StandardDataTypeEmitter.java          2     "   |  | |public| |Result|Type| |_|case|("</t>   94.7...
                        12             StandardDataTypeEmitter.java          1          "   |  | |public| |v|o|id| |_|case|("</t>   95.5...
                        13             StandardDataTypeEmitter.java          1  "|Gener|ating| |multi|ple| |construct|or|s| |f...  124.7...
                        15             StandardDataTypeEmitter.java          3  "   |  | |prot|ected| |abstr|act| |Result|Type...  128.3...
                        16             StandardDataTypeEmitter.java          2  "   |  | |prot|ected| |abstr|act| |v|o|id| |_|...  134.8...
                        17             StandardDataTypeEmitter.java          1     " |x|)| |{| |_|default|(|x|)|;| |}|\|n|\n"</t>  168.9...
                        19             StandardDataTypeEmitter.java          1  " |x|)| |{| |return| |_|default|(|x|)|;| |}|\|...  187.0...
                        23             StandardDataTypeEmitter.java          1  "\n|\|n|  | |public| |abstr|act| |<|Result|Typ...  207.5...
    Zero                1              StandardDataTypeEmitter.java          1                                              0</t>    7.9...

Alternatively, ``EvaluationResult`` provides ``aggregate()`` and ``total()`` methods to look at the data in specific demensions:

.. code-block:: python

    >>> evaluation.aggregate(['TokenType']).data
                         n_samples                               example    Entropy
    TokenType
    ClosingBracket             126                                 )</t>  20.6...
    ClosingCurlyBracket         22                                 }</t>   6.0...
    Identifier                 508  em|it|Base|Class|And|Inter|faces</t>  22.2...
    KeyWord                     69                               for</t>   7.3...
    MultilineComment           280                                 /</t>  17.7...
    NonCodeChar                 55                                 @</t>   2.4...
    One                          1                                 1</t>   6.9...
    OpeningBracket             126                                 (</t>   7.1...
    OpeningCurlyBracket         22                                 {</t>   7.2...
    Operator                   252                                 .</t>   6.7...
    Semicolon                  119                                 ;</t>   6.5...
    SpecialToken                 2                             <EOF></t>  10.6...
    StringLiteral               64                          " |{|\n"</t>  51.1...
    Zero                         1                                 0</t>   7.9...

    >>> evaluation.total()
    {'n_samples': 1647, 'Entropy': 16.0...}

When evaluation is done on file or string, by default, the line of each token and its position in the line is saved.
The version of `LM-Powered <https://github.com/giganticode/lm-powered>`_ that is currently under development uses
this information to visualize entropies for each token.

.. code-block:: python

    >>> from langmodels.evaluation import evaluate_on_string
    >>> evaluation = evaluate_on_string(trained_model, 'import java.lang.collections;')
    >>> evaluation.data
                                                           n_samples           example    Entropy
    TokenType   SubtokenNumber LinePosition TokenPosition
    Identifier  1              0            1                      1          java</t>   7.7...
                                            3                      1          lang</t>   9.0...
                2              0            5                      1  collection|s</t>  19.0...
    KeyWord     1              0            0                      1        import</t>   8.0...
    NonCodeChar 1              0            2                      1             .</t>   7.8...
                                            4                      1             .</t>   2.5...
    Semicolon   1              0            7                      1             ;</t>   4.9...


Specifying evaluation options
-----------------------------

Evaluation can be customized by passing ``EvaluationOptions`` object with specified ``metrics`` and ``characteristics``.
You can also specify ``n_processes`` to use to run pre-processing and ``batch_size`` to be used for inference:

.. code-block:: python

    >>> from langmodels.evaluation import *

    >>> evaluate_on_path(trained_model, path, save_to=output_path, batch_size=3, n_processes=1, evaluation_options=EvaluationOptions(metric_names=['Entropy'], characteristics=[TokenType()]))
    2...
    >>> evaluation = EvaluationResult.from_path(output_path)
    >>> evaluation.data
                         n_samples                                            example    Entropy
    TokenType
    ClosingBracket             126                                              )</t>  20.6...
    ClosingCurlyBracket         22                                              }</t>   6.0...
    Identifier                 508                                 type|Arguments</t>  22.2...
    KeyWord                     69                                            for</t>   7.3...
    MultilineComment           280                                              /</t>  17.7...
    NonCodeChar                 55                                              .</t>   2.4...
    One                          1                                              1</t>   6.9...
    OpeningBracket             126                                              (</t>   7.1...
    OpeningCurlyBracket         22                                              {</t>   7.2...
    Operator                   252                                              .</t>   6.7...
    Semicolon                  119                                              ;</t>   6.5...
    SpecialToken                 2                                          <EOF></t>  10.6...
    StringLiteral               64  "   |  | |prot|ected| |abstr|act| |Result|Type...  51.1...
    Zero                         1                                              0</t>   7.9...



.. |PyPI| image:: https://img.shields.io/pypi/v/giganticode-langmodels?label=pip&logo=PyPI&logoColor=white
   :target: https://pypi.org/project/giganticode-langmodels
   :alt: PyPI

.. |CI| image:: https://img.shields.io/travis/giganticode/langmodels
   :target: https://travis-ci.org/giganticode/langmodels/builds
   :alt: Travis dev branch