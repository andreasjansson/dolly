from typing import Optional
from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_DIR = "/src/.hf-cache"


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        if weights is not None and weights.name == 'weights':
            # bugfix
            weights = None

        if weights is None:
            weights = "EleutherAI/gpt-j-6B"

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
            weights, device_map="auto", cache_dir=CACHE_DIR
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            weights, cache_dir=CACHE_DIR
        )

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
            choices=["beam_search", "top_p", "top_k"],
            default="beam_search",
        ),
        num_beams: int = Input(
            description="Valid if you choose beam_search. Number of beams for beam search. 1 means no beam search.",
            default=1,
        ),
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
            default=1,
        ),
    ) -> str:
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        do_sample = False if decoding == "beam_search" else False

        outputs = self.model.generate(
            input,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p if decoding == "top_p" else 1,
            top_k=top_k if decoding == "top_k" else 50,
            repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
