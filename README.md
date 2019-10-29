# langmodels

[![Build Status](https://travis-ci.org/giganticode/langmodels.svg?branch=master)](https://travis-ci.org/giganticode/langmodels)

**Applying machine learning to large source code corpora**

# Quick start

Python version >= 3.6 required!

### Building from source

```
git clone https://github.com//giganticode/langmodels
cd langmodels
python -m venv <venv_name>
source <venv_name>/bin/activate
pip install -r requirements.txt
```

### Loading a default pre-trained model
```python
>>> import langmodels.modelregistry as reg
>>> trained_model = reg.load_default_model()

2019-10-29 12:01:21,699 [langmodels.modelregistry] INFO: Model is not found in cache. Downloading from https://www.inf.unibz.it/~hbabii/pretrained_models/langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344 ...
2019-10-29 12:01:35,732 [langmodels.model] DEBUG: Loading model from: /home/hlib/.local/share/langmodels/0.0.1/modelzoo/langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344/best.pth ...
2019-10-29 12:01:36,103 [langmodels.model] DEBUG: Using GPU for inference
```

# More model loading options

**First, all available pre-trained LMs can be listed**
```python
>>> import langmodels.modelregistry as reg
>>> reg.list_pretrained_models()

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
>>> reg.query_all_models()[0]
ModelDescription(id='langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344', bpe_merges='10k', layers_config='1024/2/1024', arch='AWD_LSTM', bin_entropy=2.1455788479, training_time_minutes_per_epoch=1429, n_epochs=6, best_epoch=5, tags=['BEST', 'DEFAULT'])
```

**The model can be loaded by tag or by id**

You can specify if you want to load a model to CPU despite having cuda-supported GPU with `force_use_cpu` parameter (defaults to `False`). If cuda-supported GPU is not available, this parameter is disregarded.
```python
>>> trained_model = reg.load_model_with_tag('BEST')

2019-10-29 11:00:04,792 [langmodels.modelregistry] INFO: Model is not found in cache. Downloading from https://www.inf.unibz.it/~hbabii/pretrained_models/langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344 ...
2019-10-29 11:00:20,136 [langmodels.model] DEBUG: Loading model from: /home/hlib/.local/share/langmodels/0.0.1/modelzoo/langmodel-large-split_10k_2_1024_191007.112241_-_langmodel-large-split_10k_2_1024_191022.141344/best.pth ...
2019-10-29 11:00:25,479 [langmodels.model] DEBUG: Using GPU for inference

>>> trained_model = reg.load_model_by_id('dev_10k_1_10_190923.132328', force_use_cpu=True)

2019-10-29 11:26:12,070 [langmodels.model] DEBUG: Loading model from: /home/hlib/.local/share/langmodels/0.0.1/modelzoo/dev_10k_1_10_190923.132328/best.pth ...
2019-10-29 11:26:12,073 [langmodels.model] DEBUG: Using CPU for inference

```

Also, you can use a lower-level API to load a model by path :
```python
trained_model = reg.load_from_path('/home/hlib/.local/share/langmodels/0.0.1/modelzoo/dev_10k_1_10_190923.132328')
```

# Inference
### Computing entropies for each line of a file

#### CLI API  
In this case the default model is used.
```
~/dev/langmodels$ python langmodels/inference/entropies.py <file> [-o <output-path>] [-e <entropy_aggregator>] [-c] [-v]

positional arguments:
  <file>                           Path to file for which entropies are to be calculated.

optional arguments:
  --output-path, -o <output-path>                    Path to file to which entropies are to be written. If not specified, --verbose option is set to true. 
  --cpu, -c                                          Forse cpu usage for inference even if cuda-supported GPU is available.
  --verbose, -v                                      Write preprocessed lines and their entropies to stdout.
```

#### Python API

```python
>>> from langmodels.modelregistry import load_default_model
>>> from langmodels.evaluation import evaluate_model_on_string

>>> trained_model = load_default_model(force_use_cpu=False)
>>> entropies = evaluate_model_on_string(trained_model, 'large\ntext', metrics=['bin_entropy'])
[EvaluationResult(text='large', prep_text=['l', 'arg', 'e</t>'], prep_metadata=(set(), [0, 3]), 
results={'bin_entropy': [15.181015908834256, 7.079181519622688, 1.809404016295282]}, 
aggregated_result={'bin_entropy': 24.069601444752227}), 
EvaluationResult(text='text', prep_text=['text</t>'], prep_metadata=(set(), [0, 1]), 
results={'bin_entropy': [13.33439828654528]}, aggregated_result={'bin_entropy': 13.33439828654528})]
```

### Autocompletion (Python API)

Example

```python
>>> from langmodels.modelregistry import load_default_model

>>> trained_model = load_default_model()
>>> trained_model.feed_text('public static main() { if')

# this does not change the state of the model:
>>> predictions = trained_model.predict_next_full_token(n_suggestions=5)
>>> print(predictions)
[('(', 0.9334765834402862), ('.', 0.01540983953864937), ('=', 0.008939018331858162), (',', 0.005372771784601065), ('the', 0.00309070517292041)]

# adding more context, if the user types '(':
>>> trained_model.feed_text('(')
[('(', 0.14554535082422237), ('c', 0.018005003646104294), ('!', 0.01614662429123089)]


# if the cursor has been moved to the beginning of the file, 
# we need to reset the state of the model (make it forget the context)
>>> trained_model.reset()
>>> trained_model.predict_next_full_token(n_suggestions=5)
[('/', 0.7209196484717589), ('package', 0.27093282656897594), ('import', 0.0007366385365522241), ('.', 0.0005714365190590807), ('public', 0.0003926736567296)]

```

# Evaluation of language models

```python
>>> from langmodels.evaluation import evaluate
>>> from langmodels.modelregistry import load_model_by_id, load_default_model     

>>> baseline = load_model_by_id('langmodel-large-split_10k_1_512_190926.120146')
>>> target = load_default_model()
>>> file = '/home/hlib/dev/langmodels/text.java'

>>> evaluate(baseline_model=baselline, target_model=target, file=file)
Output file: /home/hlib/dev/langmodels/text.java.html

```

# Editional training and Transfer learning

**TBD**
