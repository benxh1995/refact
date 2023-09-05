ctransformers_mini_db = {
    "starcoder.ggmlv3": {
        "backend": "ctransformers",
        "model_path": "TheBloke/starcoder-GGML/starcoder.ggmlv3.q4_0.bin",
        "diff_scratchpad_class": "refact_scratchpads:ScratchpadSPM",
        "chat_scratchpad_class": "refact_scratchpads:ScratchpadHuggingfaceRefact",
        "model_class_kwargs": {},
        "required_memory_mb": 12000,
        "filter_caps": ["completion", "Refact"],
    },
    "codellama-7B":{
        "backend": "ctransformers",
        "model_path": "TheBloke/codellama-7B/codellama-7b.Q2_K.gguf",
        "diff_scratchpad_class": "refact_scratchpads:ScratchpadCompletion",
        "chat_scratchpad_class": None,
        "model_class_kwargs": {},
        "required_memory_mb": 6000,
        "filter_caps": ["completion"]
    }
}
