from dataclasses import dataclass, field


@dataclass
class TranslatorModelHandlerArguments:
    tx_lm_model_name: str = field(
        default="facebook/nllb-200-distilled-600M",
        metadata={
            "help": "The pretrained language model to use. Default is 'facebook/nllb-200-distilled-600M'."
        },
    )
    tx_lm_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    tx_lm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    tx_lm_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 128."
        },
    )
    tx_lm_gen_min_new_tokens: int = field(
        default=0,
        metadata={
            "help": "Minimum number of new tokens to generate in a single completion. Default is 0."
        },
    )
    # tx_lm_gen_temperature: float = field(
    #     default=0.0,
    #     metadata={
    #         "help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.0."
    #     },
    # )
    tx_lm_gen_do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."
        },
    )
    tgt_lang: str = field(
        default="en",
        metadata={
            "help": "The target language code to translate; set this to one of WHISPER_LANGUAGE_TO_LLM_LANGUAGE. Default is 'en'."
        },
    )
