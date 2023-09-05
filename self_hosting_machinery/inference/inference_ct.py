import logging
import os
import time
import traceback
from typing import Dict, Any

from ctransformers.llm import LLM
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from refact_scratchpads import ScratchpadHuggingfaceCompletion
from self_hosting_machinery.inference import InferenceBase
from self_hosting_machinery.inference import modload
from refact_scratchpads_no_gpu.stream_results import UploadProxy
from self_hosting_machinery import env

quit_flag = False
DEBUG = int(os.environ.get("DEBUG", "0"))

class CustomAutoModelForCausalLM(AutoModelForCausalLM):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, cache_dir=None, **kwargs):
        if cache_dir is None:
            cache_dir = env.DIR_WEIGHTS
            
        
        return super().from_pretrained(cache_dir + "/" + pretrained_model_name_or_path , *args, **kwargs)

class InferenceCT(InferenceBase):
    def __init__(self,
                 model_name: str,
                 model_dict: Dict[str, Any],
                 **kwargs):
        self._model_name = model_name
        self._model_dict = model_dict
        
        self._tokenizer = CustomAutoModelForCausalLM.from_pretrained(
            self._model_dict["model_path"], cache_dir=env.DIR_WEIGHTS, model_type='gguf')

        if model_dict["backend"] == "ctransformers":
            self._model = CustomAutoModelForCausalLM.from_pretrained(
                self._model_dict["model_path"], cache_dir=env.DIR_WEIGHTS,
                **self._model_dict["model_class_kwargs"], model_type='gguf')
        else:
            raise RuntimeError(f"unknown model backend {model_dict['backend']}")
        
    def _prepare_scratchpad(self, request: Dict[str, Any]):
        def logger(*args):
            if not DEBUG:
                return
            s = " ".join([str(a) for a in args])
            logging.getLogger("MODEL").info(s)

        object_type = request["object"]
        assert object_type in ["diff_completion_req", "text_completion_req", "chat_completion_req"]
        if object_type == "diff_completion_req":
            Scratchpad = modload(self._model_dict["diff_scratchpad_class"])
        elif object_type == "chat_completion_req":
            Scratchpad = modload(self._model_dict["chat_scratchpad_class"])
        else:
            Scratchpad = ScratchpadHuggingfaceCompletion

        scratchpad = Scratchpad(tokenizer=self._tokenizer, logger=logger, **request)
        T = self._tokenizer.max_len_single_sentence
        if not isinstance(T, int) or T <= 0 or T > 4096:
            T = 2048
        p = scratchpad.prompt(T)
        logger("prompt %i tokens, max_new_tokens %i" % (len(p), request["max_tokens"]))
        if len(p) == 0:
            raise RuntimeError("empty tokens prompt")

        return scratchpad, p

    def infer(self, request: Dict[str, Any], upload_proxy: UploadProxy, upload_proxy_args: Dict):
        t0 = time.time()
        request_id = request["id"]
        try:
            scratchpad, tokens_prompt = self._prepare_scratchpad(request)
            upload_proxy_args["ts_prompt"] = time.time()
            if request_id in upload_proxy.check_cancelled():
                scratchpad.finish_reason = "cancelled"
                return

            llm = LLM(self._model, self._tokenizer)

            generation_params = {
                "prompt": scratchpad.prompt_text,
                "max_new_tokens": request["max_tokens"],
                "top_p": request.get('top_p', 1.0),
                "temperature": request.get('temperature', 0.2),
            }

            
            generated_text = llm(**generation_params)

            scratchpad.update_generated_text(generated_text)
            scratchpad.finish_reason = "maxlen"
            upload_proxy_args["ts_batch_finished"] = time.time()
            upload_proxy.upload_result(
                **upload_proxy_args,
                files=[scratchpad.completion(True)],
                finish_reason=[scratchpad.finish_reason],
                generated_tokens_n=[scratchpad.generated_tokens_n],
                more_toplevel_fields=[{}],
                status="completed"
            )
        except Exception as e:
            logging.getLogger("MODEL").error(e)
            logging.getLogger("MODEL").error(traceback.format_exc())

    def lora_switch_according_to_config(self):
        pass
