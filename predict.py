import logging
import re
import subprocess
import time
import os
from collections import OrderedDict
from typing import Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import AutoTokenizer, AutoConfig

from subclass import YieldingCausalLM


CACHE_DIR = "/src/.hf-cache"
DEFAULT_MODEL = "EleutherAI/gpt-j-6B"

os.environ['COG_WEIGHTS'] = "https://pbxt.replicate.delivery/lJBc2S9TFfSiMCVS4gyjUx5vF0AgmKyJ9HwnYe2g4ZBqbjyQA/tuned_weights.tensors"

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        if weights is not None and weights.name == 'weights':
            # bugfix
            weights = None
        if weights is None:
            self.model = self.load_huggingface_model(weights=DEFAULT_MODEL)
        elif (hasattr(weights, 'filename') and 'tensors' in weights.filename) or str(weights).endswith(".tensors"):
            self.model = self.load_tensorizer(weights)
        else:
            self.model = self.load_huggingface_model(weights=weights)

        self.tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_MODEL, cache_dir=CACHE_DIR
        )

    def load_huggingface_model(self, weights=None):
        st = time.time()
        print(f'loading weights from {weights} w/o tensorizer')
        model = YieldingCausalLM.from_pretrained(
            weights, cache_dir=CACHE_DIR
        ).to("cuda:0")
        print(f'weights loaded in {time.time() - st}')
        return model
    
    def load_tensorizer(self, weights):
        st = time.time()
        weights = str(weights)
        print("loadin")
        print(weights)
        pattern = r"https://pbxt\.replicate\.delivery/([^/]+/[^/]+)"
        match = re.search(pattern, weights)
        if match:
            weights = f"gs://replicate-files/{match.group(1)}"

        print(f"deserializing weights")
        local_weights = "/src/gpt_j_tensors"
        command = f"/gc/google-cloud-sdk/bin/gcloud storage cp {weights} {local_weights}".split()
        res = subprocess.run(command)
        if res.returncode != 0:
            raise Exception(
                f"gcloud storage cp command failed with return code {res.returncode}: {res.stderr.decode('utf-8')}"
            )
        config = AutoConfig.from_pretrained(DEFAULT_MODEL)

        logging.disable(logging.WARN)
        model = no_init_or_tensor(
            lambda: YieldingCausalLM.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )
        logging.disable(logging.NOTSET)

        des = TensorDeserializer(local_weights, plaid_mode=True)
        des.load_into_module(model)
        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Input Prompt."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        decoding: str = Input(
            description="Choose a decoding method",
            choices=["top_p", "top_k"],
            default="top_p",
        ),
        # num_beams: int = Input(
        #     description="Valid if you choose beam_search. Number of beams for beam search. 1 means no beam search.",
        #     default=1,
        # ),
        top_k: int = Input(
            description="Valid if you choose top_k decoding. The number of highest probability vocabulary tokens to keep for top-k-filtering",
            default=50,
        ),
        top_p: float = Input(
            description="Valid if you choose top_p decoding. When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1.2,
        ),
    ) -> ConcatenateIterator[str]:
        if not prompt.endswith("### Response:\n"):
            prompt += " ### Response:\n"

        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

        do_sample = False if decoding == "beam_search" else True
        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                    input,
                    max_length=max_length,
                    do_sample=do_sample,
                    #num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p if decoding == "top_p" else 1,
                    top_k=top_k if decoding == "top_k" else 50,
                    repetition_penalty=repetition_penalty,
            ):
                cur_id = output.item()
                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)

                # skip initial newline, which this almost always yields. hack - newline id = 13.
                if not first_token_yielded and not prev_ids and cur_id == 13:
                    continue

                # Words start with Ġ
                if cur_token.startswith("Ġ"):
                    if prev_ids:
                        yield self.tokenizer.decode(prev_ids)
                        prev_ids = []

                    prev_ids = [cur_id]
                    continue

                # Compound words are denoted by a "Ċ" followed by any
                # number of tokens
                if cur_token == "Ċ":
                    if prev_ids:
                        yield self.tokenizer.decode(prev_ids)
                        prev_ids = []
                    continue

                # End token
                if cur_token == "###":
                    break

                prev_ids.append(cur_id)

            if prev_ids:
                yield self.tokenizer.decode(prev_ids)
                prev_ids = []
