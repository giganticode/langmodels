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

### Getting pretrained models

```
git clone https://github.com//giganticode/modelzoo
```

set `MODEL_ZOO_PATH` env variable to point to the cloned repo, e.g.:
```
export MODEL_ZOO_PATH="$HOME/dev/modelzoo"
```

# Computing entropies for each line of a file

### CLI API  
```
~/dev/langmodels$ python langmodels/inference/entropies.py <file> [-o <output-path>] [-e <entropy_aggregator>] [-c] [-v]

positional arguments:
  <file>                           Path to file for which entropies are to be calculated.

optional arguments:
  --output-path, -o <output-path>                    Path to file to which entropies are to be written. If not specified, --verbose option is set to true. 
  --cpu, -c                                          Forse cpu usage for inference even if cuda-supported GPU is available.
  --verbose, -v                                      Write preprocessed lines and their entropies to stdout.
```

### Python API

```python
from langmodels.model import TrainedModel
from langmodels.inference.entropies import get_entropy_for_each_line

trained_model = TrainedModel.get_default_model()
entropies = get_entropy_for_each_line(trained_model, '/path/to/file')
```

# Autocompletion (Python API)

Example

```python
from langmodels.model import TrainedModel

trained_model = TrainedModel.get_default_model()
trained_model.feed_text('public static main() { if')

# this does not change the state of the model:
predictions = trained_model.predict_next_full_token(n_suggestions=5)
print(predictions)
>> [('(', 0.9334765834402862), ('.', 0.01540983953864937), ('=', 0.008939018331858162), (',', 0.005372771784601065), ('the', 0.00309070517292041)]

# adding more context, if the user types '(':
trained_model.feed_text('(')
>> [('(', 0.14554535082422237), ('c', 0.018005003646104294), ('!', 0.01614662429123089)]


# if the cursor has been moved to the beginning of the file, 
# we need to reset the state of the model (make it forget the context)
trained_model.reset()
trained_model.predict_next_full_token(n_suggestions=5)
>> [('/', 0.7209196484717589), ('package', 0.27093282656897594), ('import', 0.0007366385365522241), ('.', 0.0005714365190590807), ('public', 0.0003926736567296)]

```

# Evaluation of language models

```python
>>> from langmodels.model import TrainedModel
>>> from langmodels.model import TrainedModel

>>> medium = TrainedModel.load_model_by_name('langmodel-large-split_10k_1_512_190926.120146')
>>> small = TrainedModel.get_small_model()
>>> file = '/home/hlib/dev/langmodels/text.java'

>>> evaluate(baseline_model=small, target_model=medium, file=file)
Output file: /home/hlib/dev/langmodels/text.java.html

```