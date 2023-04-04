import os
import sys
import logging
from subprocess import call
from datetime import datetime
import torch
from tensorizer import TensorSerializer
from cog import Input, Path, BaseModel
from transformers import AutoModelForCausalLM

from training.trainer import load_training_dataset, load_tokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

MODEL_OUT = "/src/tuned_weights.tensors"


class TrainingOutput(BaseModel):
    weights: Path


def train(
    train_data: Path = Input(description="Path to data file to use for fine-tuning your model"),
    epochs: int = Input(description="Number of epochs to train", default=1),
    max_steps: int = Input(description="Maximum number of training steps", default=-1),
    ) -> TrainingOutput:
    input_model = os.environ.get("COG_WEIGHTS")
    if input_model is None:
        input_model = "EleutherAI/gpt-j-6B"

    load_training_dataset(str(train_data))
    load_tokenizer(input_model)

    root_path = os.getcwd()
    deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")

    output_dir = os.path.join("/tmp/dolly-training")
    os.makedirs(output_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    num_gpus_flag = f"--num_gpus={num_gpus}"

    print(f"Local Output Dir: {output_dir}")
    print(f"Number of GPUs: {num_gpus}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = "/src/.hf-cache"

    # TODO: use deepspeed's python api instead of subprocessing
    call("deepspeed "
         + num_gpus_flag
         + " --module training.trainer --deepspeed "
         + deepspeed_config
         + f" --training-dataset={str(train_data)}"
         + f" --input-model={input_model}"
         + f" --epochs={epochs}"
         + f" --max-steps={max_steps}"
         + " --local-output-dir "
         + output_dir
         + " --per-device-train-batch-size 8 --per-device-eval-batch-size 8 --lr 1e-5", shell=True)

    if os.path.exists(MODEL_OUT):
        os.remove(MODEL_OUT)

    model = AutoModelForCausalLM.from_pretrained(output_dir).to('cuda')
    serializer = TensorSerializer(MODEL_OUT)
    serializer.write_module(model)
    serializer.close()

    return TrainingOutput(weights=Path(MODEL_OUT))
