# langmodels
Applying machine learning to large source code corpora

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
~/dev/langmodels$ python langmodels/inference/entropies.py <file> [--output-path <output-path>] [--verbose]

positional arguments:
  <file>                           Path to file for which entropies are to be calculated.

optional arguments:
  --output-path <output-path>      Path to file to which entropies are to be written. If not specified, --verbose option is set to true. 
  --verbose                        Write preprocessed lines and their entropies to stdout.
```
