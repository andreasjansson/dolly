import os
import sys
import logging
from subprocess import call
from datetime import datetime
import torch
from cog import Input, Path, BaseModel

from training.trainer import load_training_dataset, load_tokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)


class TrainingOutput(BaseModel):
    weights: Path


def train(
    train_data: Path = Input(description="path to data file to use for fine-tuning your model"),
    ) -> TrainingOutput:
    input_model = os.environ.get("COG_WEIGHTS")
    if input_model is None:
        input_model = "EleutherAI/gpt-j-6B"

    load_training_dataset(str(train_data))
    load_tokenizer(input_model)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_name = "dolly"
    checkpoint_dir_name = f"{model_name}__{timestamp}"

    root_path = os.getcwd()
    deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")

    local_training_root = "/src/dolly_training"

    os.makedirs(local_training_root, exist_ok=True)

    local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
    dbfs_output_dir = os.path.join("/dbfs/dolly_training", checkpoint_dir_name)

    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus=}")  # TODO(andreas): remove debug

    num_gpus_flag = f"--num_gpus={num_gpus}"

    tensorboard_display_dir = f"{local_output_dir}/runs"

    print(f"Local Output Dir: {local_output_dir}")
    print(f"DBFS Output Dir: {dbfs_output_dir}")
    print(f"Tensorboard Display Dir: {tensorboard_display_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = "/src/.hf-cache"

    call("deepspeed "
         + num_gpus_flag
         + " --module training.trainer --deepspeed "
         + deepspeed_config
         + f" --training-dataset={str(train_data)}"
         + f" --input-model={input_model}"
         + " --epochs 1 --local-output-dir "
         + local_output_dir
         + " --dbfs-output-dir "
         + dbfs_output_dir
         + " --per-device-train-batch-size 8 --per-device-eval-batch-size 8 --lr 1e-5", shell=True)
