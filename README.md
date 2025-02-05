This repository is an implementation of a fine-tunable [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

The model can be fine tuned using `cog train`, and can serve predictions using `cog predict`. 

### Fine-tuning

All that `cog train` requires is an input dataset consisting of a JSON list where each example has a 'prompt' and 'completion' field. The model will be fine-tuned to produce 'completion' given 'prompt'. Here's an example command to train the model from the root directory:

```
cog train -i train_data="https://storage.googleapis.com/dan-scratch-public/fine-tuning/70k_samples_prompt.jsonl" -i gradient_accumulation_steps=8 -i learning_rate=2e-5 -i num_train_epochs=3 -i logging_steps=2 -i train_batch_size=4
```

Of the params above for training, the only required param is the `train_data`, but you can pass other parameters to modify training the model as you see fit. See the 'examples' folder for an example dataset.

### Inference

To generate text given input prompts, simply run the `cog predict` command below:
```
cog predict -i prompt="Q: Answer the following yes/no question by reasoning step-by-step. Can a dog drive a car?"
```

Note that the first prediction run will download weights for the selected model from Huggingface to a local directory; subsequent predictions will be faster. 

### Notes

This codebase was adapted from https://github.com/databrickslabs/dolly.