# Dolly

This adapted from https://github.com/databrickslabs/dolly to train Dolly with your own machine.

First, set up the environment
```
conda create -n dolly python=3.8
conda activate dolly
pip install -r requirements.txt
```

Then train with 
```python
python train_dolly.py <num_gpus>
```
For 4 x 80G A100 GPUs, the training takes about 1 hour. The trained model is saved in `dolly_training`.

Afterwards, you can load the models as follows.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "dolly_training/..."
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```