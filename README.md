# langmodels
Applying machine learning to large source code corpora

# Quick start

Python version >= 3.6 required!

## Building from source

```
git clone https://github.com//giganticode/langmodels
cd langmodels
python -m venv <venv_name>
source <venv_name>/bin/activate
pip install -r requirements.txt
```

# Computing entropies for each line of a file
```
python langmodels/inference/entropies.py <path-to-file> --path-to-output <path-to-output> --verbose
```
