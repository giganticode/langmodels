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
python langmodels/inference/entropies.py <path-to-file> --path-to-output <path-to-output> --verbose
```
