import json

from typing import Dict, List

from refact_scratchpads_no_gpu.gpt_toolbox.gpt_utils import msg, find_substring_positions
from refact_scratchpads_no_gpu.gpt_toolbox.gpt_toolbox_spad import ScratchpadToolboxGPT


class GptBugsHighlight(ScratchpadToolboxGPT):
    def __init__(self, model_n="gpt3.5-turbo-0301", supports_stream=False, **kwargs):
        super().__init__(
            model_n=model_n,
            supports_stream=supports_stream,
            **kwargs
        )

    def _messages(self) -> List[Dict[str, str]]:
        return [
            msg(
                'system',
                "You are an AI programming assistant.\n"
                "Follow the user's requirements carefully & to the letter."
            ),
            msg('user', '''
You are a code reviewer. Follow my instructions carefully & to the letter.

You are to receive a single code file.
It contain imports from other files that are present in the project, but you cannot see them.
That's why you must not highlight errors that are connected to the imports, it's a false positive.

Your assignment is:
1. Carefully read code line by line up to the end.
2. Find all errors likely to happen in runtime (ignore the imports)
3. For each found error output a comment in the following format:
{"code": "    def _messages(self) -> list[dict[str, str]]:", "description": "errors in type annotations"}
{"code": "for call, idx in enumerate(calls_unfiltered):", "description": "Invalid variable assignment"}

FIELDS DESCRIPTION:
- code: the code you found issue in
- description: brief description of the issue and short instructions hints how to fix it

Guidelines:
Explain yourself as briefly as possible, do not explain outside of code block.
The output you provide must be decodable using jsonlines format.
Do not highlight any errors connected to imports.
'''
            ),
        msg(
            'user',
            """from routers import FindRouter

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
"""
        ),
        msg(
            'assistant',
            """{"code": "from routers import FindRouter", "description": "ModuleNotFoundError: no module named routers"}"""
        ),
        msg(
            'user',
            'Not valid. You have been told to ignore any kind of import errors!'
        ),
        msg('user', self._txt)
        ]

    def _postprocess(self, completion: str) -> str:
        self.debuglog(f'Completion:\n{completion}')
        suggestions = []
        for line in completion.splitlines():
            if not line.strip():
                continue
            try:
                suggestions.append(json.loads(line))
            except Exception as e:
                self.debuglog(e)
        for s in suggestions:
            code = s['code']
            indexes = find_substring_positions(code, self._txt)
            if not indexes:
                self.debuglog('Substring not found')
                continue
            s_start, s_end = indexes
            self._txt = \
                self._txt[:s_start] + \
                f'\n<BUG>' \
                f'\nDESC: {s["description"]}\n' \
                f'{self._txt[s_start:s_end]}' \
                f'\n</BUG>' + \
                self._txt[s_end:]
        return self._txt


class GptBugsHighlightGPT4(GptBugsHighlight):
    def __init__(self, **kwargs):
        super().__init__(
            model_n='gpt-4',
            supports_stream=False,
            **kwargs
        )
