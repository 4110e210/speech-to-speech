from threading import Thread
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import torch

from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
from nltk import sent_tokenize

# import re

logger = logging.getLogger(__name__)

console = Console()

# https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/special_tokens_map.json
WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "eng_Latn", #"english"
    "fr": "fra_Latn", #"french"
    "es": "spa_Latn", #"spanish"
    "zh": "zho_Hans", #"chinese"
    "ja": "jpn_Jpan", #"japanese"
    "ko": "kor_Hang", #"korean"
    "hi": "hin_Deva", #"hindi"
    "de": "deu_Latn", #"german"
    "pt": "por_Latn", #"portuguese"
    "pl": "pol_Latn", #"polish"
    "it": "ita_Latn", #"italian"
    "nl": "nld_Latn", #"dutch"
    "vi": "vie_Latn", # Vietnam
    "id": "zsm_Latn", # Indonesia
}

class TranslatorModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="facebook/nllb-200-distilled-600M",
        device="cuda",
        torch_dtype="float16",
        tgt_lang="en",
        gen_kwargs={},
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        # self.tgt_lang=tgt_lang
        
        self.pipe = pipeline(
            'translation', model=self.model, tokenizer=self.tokenizer, device=device,
            src_lang="zho_Hans", tgt_lang=WHISPER_LANGUAGE_TO_LLM_LANGUAGE[tgt_lang], # max_length=400,
        )
        
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            **gen_kwargs,
        }

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_chat = "您好！"
        warmup_gen_kwargs = {
            # "src_lang":"zh", 
            # "tgt_lang":"en", 
            
            **self.gen_kwargs,
        }

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        n_steps = 1

        for _ in range(n_steps):
            thread = Thread(
                target=self.pipe, args=(dummy_chat,), kwargs=self.gen_kwargs #warmup_gen_kwargs
            )
            thread.start()
            generated_text = ""
            for new_text in self.streamer:
                generated_text += new_text
                # pass
            logger.info(
                f"{self.__class__.__name__}:  warmed up! dummy_chat: {generated_text}"
            )

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, prompt):
        logger.debug("infering translator model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]

        thread = Thread(
            target=self.pipe, args=(prompt,), kwargs=self.gen_kwargs
        )
        thread.start()
        if self.device == "mps":
            generated_text = ""
            for new_text in self.streamer:
                generated_text += new_text
            printable_text = generated_text
            torch.mps.empty_cache()
        else:
            generated_text, printable_text = "", ""
            for new_text in self.streamer:
                generated_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    yield (sentences[0], language_code)
                    printable_text = new_text

        # don't forget last sentence
        yield (printable_text, language_code)
