"""Prompt transport independent of the OS argument and environment limits."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile


# Native protocols remain in their adapters. Runtime-file is only for CLIs
# whose one-shot parser accepts text exclusively as an argument.
TRANSPORTS = {
    "codex": "protocol",
    "claude": "stdin",
    "opencode": "stdin",
    "cursor": "stdin",
    "agy": "json-stdin",
    "devin": "file",
    "grok": "file",
    "dsh": "file",
    "zcode": "runtime-file",
    "hermes": "runtime-file",
}


def stage_prompt(directory: Path, run_id: str, prompt: str) -> Path:
    path = directory / f".aop-prompt-{run_id}.txt"
    fd, temporary = tempfile.mkstemp(prefix=".prompt-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(prompt)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def stdin_prompt(provider: str, prompt: str) -> str | None:
    mode = TRANSPORTS[provider]
    if mode == "stdin":
        return prompt
    if mode == "json-stdin":
        return (
            json.dumps({"event": "user", "message": {"role": "user", "content": prompt}})
            + "\n"
        )
    return None


@contextmanager
def prompt_stream(prompt: str | None):
    """A seekable stdin avoids blocking on pipe writes before the timeout starts."""
    if prompt is None:
        yield None
        return
    with tempfile.TemporaryFile() as stream:
        stream.write(prompt.encode("utf-8"))
        stream.seek(0)
        yield stream


# These execute inside the native interpreter, before its CLI parser. No exec
# crosses the OS boundary after the prompt is restored to the runtime's argv.
PYTHON_ARGV_LOADER = """import pathlib, runpy, sys
sys.argv = sys.argv[1:]
flag = '--prompt' if '--prompt' in sys.argv else '-q'
i = sys.argv.index(flag) + 1
sys.argv[i] = pathlib.Path(sys.argv[i]).read_bytes().decode('utf-8')
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name='__main__')
"""
NODE_ARGV_LOADER = """const fs = require('node:fs');
const i = process.argv.indexOf('--prompt') + 1;
if (!i) throw new Error('AOP prompt argument missing');
process.argv[i] = fs.readFileSync(process.argv[i], 'utf8');
"""
