## **langmodels**

[![Build Status](https://travis-ci.org/giganticode/langmodels.svg?branch=master)](https://travis-ci.org/giganticode/langmodels)

This is a repository for **neural language models (LMs)** trained on a large corpus of source code 
and a toolkit to work with such models. 

You could be interested in using this library if you want to:
* Use existing pre-trained models for tasks such as autocompletion and bug prediction;
* Use the pre-trained models for transfer transfer learning or further fine-tuning;
* Train a model from scratch by choosing one of the wide range of corpus preprocessing choices, 
 neural network (NN) architectures, and training options.

This project uses [fastai](https://www.fast.ai) and 
[pytorch](https://pytorch.org) libraries for NN training/inference. 
For corpus preprocessing [giganticode-dataprep](https://github.com/giganticode/dataprep) is used.

## Quick start

### Prerequisites

* Python version >= 3.6 required! 

### Installation

```shell script
pip install giganticode-langmodels
```

OR to build from source:

```
git clone https://github.com//giganticode/langmodels
cd langmodels
python -m venv langmodels-venv
source langmodels-venv/bin/activate
pip install -r requirements.txt
```

## Using existing pre-trained models
### Loading a default pre-trained model
```python
>>> import langmodels.repository as repo
>>> trained_model = repo.load_default_model()

[langmodels.repository] INFO: Model is not found in cache. Downloading from https://www.inf.unibz.it/~hbabii/pretrained_models/langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344 ...
[langmodels.model] DEBUG: Loading model from: /home/hlib/.local/share/langmodels/0.0.1/modelzoo/langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344/best.pth ...
[langmodels.model] DEBUG: Using GPU for inference
```

### Other model loading options

**To see which models are available, you can call `list_pretrained_models` function.**

Set `cached` parameter to `True` (default is `False`) to display only cached LMs (e.g. if offline).
```python
>>> import langmodels.repository as repo
>>> repo.list_pretrained_models(cached=False)

  ID                                                                    BPE_MERGES  LAYERS_CONFIG  ARCH      BIN_ENTROPY    TRAINING_TIME_MINUTES_PER_EPOCH  N_EPOCHS  BEST_EPOCH  TAGS                 
    
  langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-spl  10k         1024/2/1024    AWD_LSTM  2.1455788479   1429                             6         5           ['BEST', 'DEFAULT']  
  it_10k_2_1024_191022.141344                                                                                                                                                                           
  langmodel-large-split_10k_3_1024_191007.112257_-_langmodel-large-spl  10k         512/3/1024     AWD_LSTM  2.14730056622  1432                             6         5           []                   
  it_10k_3_1024_191022.134822                                                                                                                                                                           
  langmodel-large-split_10k_2_2048_191007.112249_-_langmodel-large-spl  10k         512/2/2048     GRU       2.19923468325  1429                             6         5           []                   
  it_10k_2_2048_191022.141335                                                                                                                                                                           
  langmodel-large-split_10k_1_512_190926.120146                         10k         512/1/512      AWD_LSTM  2.69019493253  479                              9         8           ['MEDIUM']           
  langmodel-small-split_10k_1_512_190906.154943                         10k         512/1/512      AWD_LSTM  4.73768141172  4                                19        18          ['TINY']             
  dev_10k_1_10_190923.132328                                            10k         10/1/10        AWD_LSTM  9.15688191092  0                                0         -1          ['RANDOM']
```

Use `query_all_models` method to get a list of `ModelDescription` objects
```python
>>> import langmodels.repository as repo
>>> repo.query_all_models()[0]
ModelDescription(id='langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344', bpe_merges='10k', layers_config='1024/2/1024', arch='AWD_LSTM', bin_entropy=2.1455788479, training_time_minutes_per_epoch=1429, n_epochs=6, best_epoch=5, tags=['BEST', 'DEFAULT'])
```

**A model can be loaded by tag or by id.**

You can specify if you want to load a model to CPU despite having cuda-supported GPU with `force_use_cpu` parameter 
(defaults to `False`). If cuda-supported GPU is not available, this parameter is disregarded.
```python
>>> trained_model = repo.load_model_with_tag('BEST')

>>> trained_model = repo.load_model_by_id('dev_10k_1_10_190923.132328_new', force_use_cpu=True)
```

Also, you can use a lower-level API to load a model by path :
```python
trained_model = repo.load_from_path('/home/hlib/.local/share/langmodels/0.0.1/modelzoo/dev_10k_1_10_190923.132328_new')
```

## Inference
### Autocompletion

Example

```python
>>> import langmodels.repository as repo
>>> trained_model = repo.load_default_model()
>>> trained_model.feed_text('public static main() { if', extension='java')

# this does not change the state of the model:
>>> predictions = trained_model.predict_next_full_token(n_suggestions=5)
[('(', 0.9334765834402862), ('.', 0.01540983953864937), ('=', 0.008939018331858162), (',', 0.005372771784601065), ('the', 0.00309070517292041)]

# adding more context:
>>> trained_model.feed_text('(', extension='java')
>>> trained_model.predict_next_full_token(n_suggestions=3)
[('(', 0.14554535082422237), ('c', 0.018005003646104294), ('!', 0.01614662429123089)]


# resetting the state of the model (make it forget the context)
>>> trained_model.reset()
>>> trained_model.predict_next_full_token(n_suggestions=5)
[('/', 0.7209196484717589), ('package', 0.27093282656897594), ('import', 0.0007366385365522241), ('.', 0.0005714365190590807), ('public', 0.0003926736567296)]

```


### Bug prediction based on per-line entropies evaluation

An LM can be used to calculate cross-entropies for each line of a file. High values can give an idea about 
unusual/suspicious chunks of code [[1]](#1).

Check section [LM Evaluation](#lm-evaluation) section to learn how to calculate 
cross-entropy for a project/file/string,

Check our [vsc plugin](https://github.com/giganticode/vsc-extension) for highlighting suspicious code.

## Fine-tuning and Transfer learning

**TBD**

## Training from scratch (Not supported on OSx)

### Python API

```python
>>> from langmodels.training.training import train
>>> from langmodels.lmconfig.datamodel import *

>>> train(LMTrainingConfig(corpus=Corpus(path='/path/to/the/dataset')))
```

More parameters to customize corpus pre-processing, NN architecture, and the training process can be specified:

```python
>>> from langmodels.training.training import train
>>> from langmodels.lmconfig.datamodel import *

>>> train(LMTrainingConfig(corpus=Corpus(path='/path/to/the/dataset'), 
                            prep_function=PrepFunction(options=PrepFunctionOptions(no_com=False, no_unicode=True)),
                            arch=GRUArchj(n_layers=2),
                            training=Training(weight_decay=5e-6)
))
```

Below you can see all the default parameters specified explicitly:

```python
>>> from langmodels.lmconfig.datamodel import *
>>> from langmodels.training.training import train

>>> train(LMTrainingConfig(base_model=None, 
                       bs=32, 
                       corpus=Corpus(path=os.path.join(HOME, 'dataset'), extensions="java"), 
                       prep_function=PrepFunction(corpus_api.bpe, ['10k'], 
                                                  PrepFunctionOptions(no_com=False, no_unicode=True, 
                                                                    no_spaces=True, max_str_length=sys.maxsize)), 
                       arch=LstmArch(
                           bidir=False, qrnn=False, emb_sz=1024, n_hid=1024, n_layers=3, 
                           drop=Dropouts(multiplier=0.5, oute=0.02, outi=0.25, outh=0.15, w=0.2, out=0.1), 
                           tie_weights=True, out_bias=True), 
                       bptt=200, 
                       training=Training(
                            optimizer=Adam(betas=(0.9, 0.99)),
                            files_per_epoch=50000,
                            gradient_clip=0.3,
                            activation_regularization=ActivationRegularization(alpha=2., beta=1.), 
                            schedule=RafaelsTrainingSchedule(init_lr=1e-4, mult_coeff=0.5, patience=0,
                                                            max_epochs=50, max_lr_reduction_times=6), 
                            weight_decay=1e-6)
                       )
      )
```

### CLI API

Training can be run from command line as simple as running `train` command passing path to the config in json format 
as `--config` param. To override values in the json file (or default values if `--config` param is not specified), 
you can use `--patch` param.
```shell script
>> langmodels train --config="/path/to/json/config.json" --patch="bs=64,arch.drop.multiplier=3.0"
```

If neither `--config` nor `--patch` params are specified, the training will be running with the default parameters.
The json with the default parameters would look like follows:

```json
{'arch': {'bidir': False,
          'drop': {'multiplier': 0.5,
                   'out': 0.1,
                   'oute': 0.02,
                   'outh': 0.15,
                   'outi': 0.25,
                   'w': 0.2},
          'emb_sz': 1024,
          'n_hid': 1024,
          'n_layers': 3,
          'name': 'lstm',
          'out_bias': True,
          'qrnn': False,
          'tie_weights': True},
 'base_model': None,
 'bptt': 200,
 'bs': 32,
 'config_version': '0.0.3-alpha.0',
 'corpus': {'extensions': 'java', 'path': '/Users/hlib/dataset'},
 'prep_function': {'callable': 'bpe',
                   'options': {'max_str_length': 9223372036854775807,
                               'no_com': False,
                               'no_spaces': True,
                               'no_str': False,
                               'no_unicode': True},
                   'params': ['10k']},
 'training': {'activation_regularization': {'alpha': 2.0, 'beta': 1.0},
              'files_per_epoch': 50000,
              'gradient_clip': 0.3,
              'optimizer': {'betas': [0.9, 0.99], 'name': 'Adam'},
              'schedule': {'init_lr': 0.0001,
                           'max_epochs': 50,
                           'max_lr_reduction_times': 6,
                           'mult_coeff': 0.5,
                           'name': 'rafael',
                           'patience': 0},
              'weight_decay': 1e-06}}
```

Most probably, you would have to override at least the `corpus.path` value.

For more options, run:
```shell script
>> langmodels train --help
```

## LM Evaluation

When training a language model, it is important to be able to evaluate LM's performance.
In this section we describe different ways to do this using `langmodels` library. 
You can also use our [tool](https://github.com/giganticode/lm-powered) to visualize the evaluation.

### Evaluation on a string / file

First, a model can be evaluate on a string with `evaluate_model_on_string` method. Note that the result may differ a lot depending 
on the state of the model. Use methods `reset` and `feed_text` to reset the model 
to initial state and change the context of the model respectively.

```python

>>> import langmodels.repository as repo 
>>> from langmodels.evaluation import evaluate_model_on_string    

>>> model = repo.load_default_model()
>>> evaluate_model_on_string(model, 'public class MyClass {')

{full_token_entropy/ParsedToken: EvaluationResult(
    tokens=['public</t>', 'class</t>', 'MyClass</t>', '{</t>'],
    token_types=['KeyWord', 'KeyWord', 'SplitContainer', 'OpeningCurlyBracket'],
    values=[1.8144783973693848, 3.668722629547119, 0.5620064437389374, 0.2571456730365753], 
    aggregated_value=1.5755882859230042
)}

```

Similarly, `evaluate_model_on_file` will return a list of `Evaluation` object (1 per each line)

### Evaluation on a corpus

Evaluation can be run on a set of files with `evaluate_model_on_path` method

```python
>>> import langmodels.repository as repo 
>>> from langmodels.evaluation import evaluate_model_on_path

>>> model = repo.load_default_model()
>>> evaluate_model_on_path(model, '/path/to/file')

100%|████████████████████████████████████████████████████████████████████████████| 28/28 [00:11<00:00,  2.35it/s]
{full_token_entropy/ParsedToken: (5.859160765187885, 5745)}
```

In `full_token_entropy/ParsedToken`: `full_token_entropy` is a metric used to evaluate the performance; 
`ParsedToken` means that all the tokens were considered when evaluating (See the next section for more details).
Thus, the average full-token-entropy is ~ 5.85 evaluated on 5.7k tokens.

### Specifying metrics

You can specify based on which metrics the model is to be evaluated.

```python
>>> import langmodels.repository as repo 
>>> from langmodels.evaluation import evaluate_model_on_path

>>> model = repo.load_default_model()
>>> evaluate_model_on_path(model, '/path/to/file', metrics={'full_token_entropy', 'mrr'})
```

Possible metric values are `full_token_entropy`, `subtoken_entropy`, `mrr`. Default metric set is `{full_token_entropy}`


## Release Notes

### 0.0.3-alpha.0 (NOT backward-compatible with 0.0.1-alpha.2)

- Config datamodel improvements: 
    - Add possibility to specify SGD optimizer; 
    - Add patience param to training scedule;
    - Add converters between versions of configs;
- Training:
    - Report binary entropy instead of log-base-e entropy;
    - Save more model metrics (size on disk, trainable params, training time per epoch);
    - Do not save model after every epoch by default;
- Evaluation improvements:
    - Return token types in `EvaluationResult`;
    - Add possibility to specify token types to be considered when running evaluation;
    - Trained_model.predict_next_token(): return 1 suggestion by default;
- Add script for new models upload.

### 0.0.1-alpha.2 (NOT backward-compatible with 0.0.1-alpha.1)

- Make downloading model from the repository thread-safe
- Force to specify the extension which corresponds to the type of the code fed into
the `TrainedModel`. **API change**: `trained_model.feed_text(text: str)` -> `trained_model.feed_text(text: str, extension: str)`

### 0.0.1-alpha.1

Make methods of `TrainedModel` that change underlying PyTorch model thread-safe

### 0.0.1-alpha.0

Initial PyPI release

## References

<a id="1">[1]</a> Ray, B., Hellendoorn, V., Godhane, S., Tu, Z., Bacchelli, A., & Devanbu, P. (2016, May). 
On the" naturalness" of buggy code. In 2016 IEEE/ACM 38th International Conference on Software Engineering (ICSE) 
(pp. 428-439). IEEE.