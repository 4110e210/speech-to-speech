import logging
import os
import sys
from copy import copy
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional
from sys import platform
from VAD.vad_handler import VADHandler
from arguments_classes.chat_tts_arguments import ChatTTSHandlerArguments
from arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from arguments_classes.mlx_language_model_arguments import (
    MLXLanguageModelHandlerArguments,
)
from arguments_classes.translator_model_arguments import TranslatorModelHandlerArguments
from arguments_classes.module_arguments import ModuleArguments
from arguments_classes.paraformer_stt_arguments import ParaformerSTTHandlerArguments
from arguments_classes.parler_tts_arguments import ParlerTTSHandlerArguments
from arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from arguments_classes.socket_sender_arguments import SocketSenderArguments
from arguments_classes.vad_arguments import VADHandlerArguments
from arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from arguments_classes.faster_whisper_stt_arguments import (
    FasterWhisperSTTHandlerArguments,
)
from arguments_classes.melo_tts_arguments import MeloTTSHandlerArguments
from arguments_classes.open_api_language_model_arguments import OpenApiLanguageModelHandlerArguments
from arguments_classes.facebookmms_tts_arguments import FacebookMMSTTSHandlerArguments
import torch
import nltk
from rich.console import Console
from transformers import (
    HfArgumentParser,
)

from utils.thread_manager import ThreadManager

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

console = Console()
logging.getLogger("numba").setLevel(logging.WARNING)  # quiet down numba logs


def rename_args(args, prefix):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1 :]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


def parse_arguments():
    parser = HfArgumentParser(
        (
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            ParaformerSTTHandlerArguments,
            FasterWhisperSTTHandlerArguments,
            LanguageModelHandlerArguments,
            OpenApiLanguageModelHandlerArguments,
            MLXLanguageModelHandlerArguments,
            TranslatorModelHandlerArguments,
            ParlerTTSHandlerArguments,
            MeloTTSHandlerArguments,
            ChatTTSHandlerArguments,
            FacebookMMSTTSHandlerArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse configurations from a JSON file if specified
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse arguments from command line if no JSON file is provided
        return parser.parse_args_into_dataclasses()


def setup_logger(log_level):
    global logger
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)


def optimal_mac_settings(mac_optimal_settings: Optional[str], *handler_kwargs):
    if mac_optimal_settings:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "device"):
                kwargs.device = "mps"
            if hasattr(kwargs, "mode"):
                kwargs.mode = "local"
            if hasattr(kwargs, "stt"):
                kwargs.stt = "moonshine"
            if hasattr(kwargs, "llm"):
                kwargs.llm = "mlx-lm"
            if hasattr(kwargs, "tts"):
                kwargs.tts = "melo"


def check_mac_settings(module_kwargs):
    if platform == "darwin":
        if module_kwargs.device == "cuda":
            raise ValueError(
                "Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'."
            )
        if module_kwargs.llm != "mlx-lm":
            logger.warning(
                "For macOS users, it is recommended to use mlx-lm. You can activate it by passing --llm mlx-lm."
            )
        if module_kwargs.tts != "melo":
            logger.warning(
                "If you experiences issues generating the voice, considering setting the tts to melo."
            )


def overwrite_device_argument(common_device: Optional[str], *handler_kwargs):
    if common_device:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "lm_device"):
                kwargs.lm_device = common_device
            if hasattr(kwargs, "tts_device"):
                kwargs.tts_device = common_device
            if hasattr(kwargs, "stt_device"):
                kwargs.stt_device = common_device
            if hasattr(kwargs, "paraformer_stt_device"):
                kwargs.paraformer_stt_device = common_device
            if hasattr(kwargs, "facebook_mms_device"):
                kwargs.facebook_mms_device = common_device


def prepare_module_args(module_kwargs, *handler_kwargs):
    optimal_mac_settings(module_kwargs.local_mac_optimal_settings, module_kwargs)
    if platform == "darwin":
        check_mac_settings(module_kwargs)
    overwrite_device_argument(module_kwargs.device, *handler_kwargs)


def prepare_all_args(
    module_kwargs,
    whisper_stt_handler_kwargs,
    paraformer_stt_handler_kwargs,
    faster_whisper_stt_handler_kwargs,
    language_model_handler_kwargs,
    open_api_language_model_handler_kwargs,
    mlx_language_model_handler_kwargs,
    translator_model_handler_kwargs, # C.T.Lin
    parler_tts_handler_kwargs,
    melo_tts_handler_kwargs,
    chat_tts_handler_kwargs,
    facebook_mms_tts_handler_kwargs,
):
    prepare_module_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        mlx_language_model_handler_kwargs,
        translator_model_handler_kwargs,
        parler_tts_handler_kwargs,
        melo_tts_handler_kwargs,
        chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs,
    )

    rename_args(whisper_stt_handler_kwargs, "stt")
    rename_args(faster_whisper_stt_handler_kwargs, "faster_whisper_stt")
    rename_args(paraformer_stt_handler_kwargs, "paraformer_stt")
    rename_args(language_model_handler_kwargs, "lm")
    rename_args(mlx_language_model_handler_kwargs, "mlx_lm")
    rename_args(open_api_language_model_handler_kwargs, "open_api")
    rename_args(translator_model_handler_kwargs, "tx_lm") # C.T.Lin
    rename_args(parler_tts_handler_kwargs, "tts")
    rename_args(melo_tts_handler_kwargs, "melo")
    rename_args(chat_tts_handler_kwargs, "chat_tts")
    rename_args(facebook_mms_tts_handler_kwargs, "facebook_mms")


def initialize_queues_and_events():
    return {
        "stop_event": Event(),
        "should_listen": Event(),
        "recv_audio_chunks_queue": Queue(),
        "send_audio_chunks_queue": Queue(),
        "spoken_prompt_queue": Queue(),
        "text_prompt_queue": Queue(),
        "lm_response_queue": Queue(),
    }


def build_pipeline(
    module_kwargs,
    socket_receiver_kwargs,
    socket_sender_kwargs,
    vad_handler_kwargs,
    whisper_stt_handler_kwargs,
    faster_whisper_stt_handler_kwargs,
    paraformer_stt_handler_kwargs,
    language_model_handler_kwargs,
    open_api_language_model_handler_kwargs,
    mlx_language_model_handler_kwargs,
    translator_model_handler_kwargs, # C.T.Lin
    parler_tts_handler_kwargs,
    melo_tts_handler_kwargs,
    chat_tts_handler_kwargs,
    facebook_mms_tts_handler_kwargs,
    queues_and_events,
):
    stop_event = queues_and_events["stop_event"]
    should_listen = queues_and_events["should_listen"]
    recv_audio_chunks_queue = queues_and_events["recv_audio_chunks_queue"]
    send_audio_chunks_queue = queues_and_events["send_audio_chunks_queue"]
    spoken_prompt_queue = queues_and_events["spoken_prompt_queue"]
    text_prompt_queue = queues_and_events["text_prompt_queue"]
    lm_response_queue = queues_and_events["lm_response_queue"]
    if module_kwargs.mode == "local":
        from connections.local_audio_streamer import LocalAudioStreamer

        local_audio_streamer = LocalAudioStreamer(
            input_queue=recv_audio_chunks_queue, output_queue=send_audio_chunks_queue
        )
        comms_handlers = [local_audio_streamer]
        should_listen.set()
    else:
        from connections.socket_receiver import SocketReceiver
        from connections.socket_sender import SocketSender

        comms_handlers = [
            SocketReceiver(
                stop_event,
                recv_audio_chunks_queue,
                should_listen,
                host=socket_receiver_kwargs.recv_host,
                port=socket_receiver_kwargs.recv_port,
                chunk_size=socket_receiver_kwargs.chunk_size,
            ),
            SocketSender(
                stop_event,
                send_audio_chunks_queue,
                host=socket_sender_kwargs.send_host,
                port=socket_sender_kwargs.send_port,
            ),
        ]

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )

    stt = get_stt_handler(module_kwargs, stop_event, spoken_prompt_queue, text_prompt_queue, whisper_stt_handler_kwargs, faster_whisper_stt_handler_kwargs, paraformer_stt_handler_kwargs)
    lm = get_llm_handler(module_kwargs, stop_event, text_prompt_queue, lm_response_queue, language_model_handler_kwargs, open_api_language_model_handler_kwargs, mlx_language_model_handler_kwargs, translator_model_handler_kwargs)
    tts = get_tts_handler(module_kwargs, stop_event, lm_response_queue, send_audio_chunks_queue, should_listen, parler_tts_handler_kwargs, melo_tts_handler_kwargs, chat_tts_handler_kwargs, facebook_mms_tts_handler_kwargs)

    return ThreadManager([*comms_handlers, vad, stt, lm, tts])


def get_stt_handler(module_kwargs, stop_event, spoken_prompt_queue, text_prompt_queue, whisper_stt_handler_kwargs, faster_whisper_stt_handler_kwargs, paraformer_stt_handler_kwargs):
    if module_kwargs.stt == "moonshine":
        from STT.moonshine_handler import MoonshineSTTHandler
        return MoonshineSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
        )
    if module_kwargs.stt == "whisper":
        from STT.whisper_stt_handler import WhisperSTTHandler
        return WhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "whisper-mlx":
        from STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler
        return LightningWhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "paraformer":
        from STT.paraformer_handler import ParaformerSTTHandler
        return ParaformerSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(paraformer_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "faster-whisper":
        from STT.faster_whisper_handler import FasterWhisperSTTHandler

        return FasterWhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(faster_whisper_stt_handler_kwargs),
        )
    else:
        raise ValueError("The STT should be either whisper, whisper-mlx, or paraformer.")


def get_llm_handler(
    module_kwargs, 
    stop_event, 
    text_prompt_queue, 
    lm_response_queue, 
    language_model_handler_kwargs,
    open_api_language_model_handler_kwargs,
    mlx_language_model_handler_kwargs,
    translator_model_handler_kwargs
):
    if module_kwargs.llm == "transformers":
        from LLM.language_model import LanguageModelHandler
        return LanguageModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(language_model_handler_kwargs),
        )
    elif module_kwargs.llm == "open_api":
        from LLM.openai_api_language_model import OpenApiModelHandler
        return OpenApiModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(open_api_language_model_handler_kwargs),
        )

    elif module_kwargs.llm == "mlx-lm":
        from LLM.mlx_language_model import MLXLanguageModelHandler
        return MLXLanguageModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(mlx_language_model_handler_kwargs),
        )

    elif module_kwargs.llm == "translate":
        from LLM.translator_model import TranslatorModelHandler
        return TranslatorModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(translator_model_handler_kwargs),
        )
    
    else:
        raise ValueError("The LLM should be either transformers or mlx-lm")


def get_tts_handler(module_kwargs, stop_event, lm_response_queue, send_audio_chunks_queue, should_listen, parler_tts_handler_kwargs, melo_tts_handler_kwargs, chat_tts_handler_kwargs, facebook_mms_tts_handler_kwargs):
    if module_kwargs.tts == "parler":
        from TTS.parler_handler import ParlerTTSHandler
        return ParlerTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(parler_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "melo":
        try:
            from TTS.melo_handler import MeloTTSHandler
        except RuntimeError as e:
            logger.error(
                "Error importing MeloTTSHandler. You might need to run: python -m unidic download"
            )
            raise e
        return MeloTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(melo_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "chatTTS":
        try:
            from TTS.chatTTS_handler import ChatTTSHandler
        except RuntimeError as e:
            logger.error("Error importing ChatTTSHandler")
            raise e
        return ChatTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(chat_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "facebookMMS":
        from TTS.facebookmms_handler import FacebookMMSTTSHandler
        return FacebookMMSTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(facebook_mms_tts_handler_kwargs),
        )
    else:
        raise ValueError("The TTS should be either parler, melo, chatTTS or facebookMMS")


def main():
    (
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        vad_handler_kwargs,
        whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs,  # Add this line
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        mlx_language_model_handler_kwargs,
        translator_model_handler_kwargs, # C.T.Lin
        parler_tts_handler_kwargs,
        melo_tts_handler_kwargs,
        chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs,
    ) = parse_arguments()

    setup_logger(module_kwargs.log_level)

    prepare_all_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs,  # Add this line
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        mlx_language_model_handler_kwargs,
        translator_model_handler_kwargs, # C.T.Lin
        parler_tts_handler_kwargs,
        melo_tts_handler_kwargs,
        chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs,
    )

    queues_and_events = initialize_queues_and_events()

    pipeline_manager = build_pipeline(
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        vad_handler_kwargs,
        whisper_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs,  # Add this line
        paraformer_stt_handler_kwargs,
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        mlx_language_model_handler_kwargs,
        translator_model_handler_kwargs, # C.T.Lin
        parler_tts_handler_kwargs,
        melo_tts_handler_kwargs,
        chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs,
        queues_and_events,
    )

    try:
        pipeline_manager.start()
    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":
    main()
