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
  --entropy-aggregator, -e <entropy_aggregator>      Fuction to calculate entropy for the whole line from subtoken entropies. Possible values: 
                                                     'subtoken-average': average over all subtokens' entropies 
                                                     'full-token-average' (default): average over all full-tokens' entopies (entropy of a full token is a sum of entopies of its subtokens to which a token was split during pre-processing) 
                                                     'full-token-entropies': a list of full-token entropies (gives freedom to library's clients to compute line-entropy in their own way)
  --cpu, -c                                          Forse cpu usage for inference even if cuda-supported GPU is available.
  --verbose, -v                                      Write preprocessed lines and their entropies to stdout.
```
