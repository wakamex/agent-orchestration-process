"""unittest support for AOP's tests.

Each test runs with a private temporary directory, reversible environment and
attribute patches, optional stdout and stderr capture, and named fixtures:
a test method's parameters are filled by fixtures registered with @fixture,
which may themselves take fixtures, or by @parametrize values.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import io
import os
import re
import shutil
import stat
import sys
import tempfile
import types
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

_FIXTURES: dict[str, tuple[Callable[..., Any], bool]] = {}


def fixture(function: Callable[..., Any] | None = None, *, autouse: bool = False):
    """Register a fixture under its function name; generators run teardown code."""

    def register(found: Callable[..., Any]) -> Callable[..., Any]:
        _FIXTURES[found.__name__] = (found, autouse)
        return found

    return register(function) if function is not None else register


def parametrize(names: str | tuple[str, ...] | list[str], values: list[Any]):
    """Run the test once per value; stacked decorators combine their values."""
    keys = (
        [name.strip() for name in names.split(",") if name.strip()]
        if isinstance(names, str)
        else list(names)
    )

    def decorate(test: Callable[..., Any]) -> Callable[..., Any]:
        rows = [
            dict(zip(keys, value if len(keys) > 1 else (value,), strict=True))
            for value in values
        ]
        test._parameters = [  # type: ignore[attr-defined]
            {**outer, **inner}
            for inner in rows
            for outer in getattr(test, "_parameters", [{}])
        ]
        return test

    return decorate


def mark(name: str):
    """Label a test, for fixtures that check `request.marks`."""

    def decorate(test: Callable[..., Any]) -> Callable[..., Any]:
        test._marks = {*getattr(test, "_marks", set()), name}  # type: ignore[attr-defined]
        return test

    return decorate


def skipif(condition: bool, *, reason: str):
    return unittest.skipIf(condition, reason)


def skip(reason: str) -> None:
    raise unittest.SkipTest(reason)


def fail(message: str) -> None:
    raise AssertionError(message)


@contextlib.contextmanager
def raises(
    expected: type[BaseException] | tuple[type[BaseException], ...],
    match: str | None = None,
) -> Iterator[types.SimpleNamespace]:
    """Require the block to raise `expected`, with `match` found in its message."""
    info = types.SimpleNamespace(value=None)
    try:
        yield info
    except expected as error:
        if match is not None and re.search(match, str(error)) is None:
            raise AssertionError(f"{match!r} not found in {str(error)!r}") from error
        info.value = error
        return
    raise AssertionError(f"did not raise {expected}")


class MonkeyPatch:
    """Environment, attribute, and working-directory changes undone after a test."""

    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []

    def setenv(self, name: str, value: str) -> None:
        self._remember_env(name)
        os.environ[name] = str(value)

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        self._remember_env(name)
        del os.environ[name]

    def setattr(self, target: Any, name: str, value: Any) -> None:
        if not hasattr(target, name):
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        # A class's own __dict__ entry keeps descriptors such as staticmethod;
        # an inherited attribute is removed again rather than copied down.
        own = isinstance(target, type) and name in vars(target)
        previous = vars(target)[name] if own else getattr(target, name)
        setattr(target, name, value)
        if isinstance(target, type) and not own:
            self._undo.append(lambda: delattr(target, name))
        else:
            self._undo.append(lambda: setattr(target, name, previous))

    def chdir(self, path: str | os.PathLike[str]) -> None:
        previous = os.getcwd()
        os.chdir(path)
        self._undo.append(lambda: os.chdir(previous))

    def undo(self) -> None:
        while self._undo:
            self._undo.pop()()

    def _remember_env(self, name: str) -> None:
        previous = os.environ.get(name)
        self._undo.append(
            lambda: os.environ.pop(name, None)
            if previous is None
            else os.environ.__setitem__(name, previous)
        )


class CaptureResult(NamedTuple):
    out: str
    err: str


class Capture:
    """Captured stdout and stderr; `readouterr` returns and clears what was written."""

    def __init__(self) -> None:
        self._out, self._err = io.StringIO(), io.StringIO()
        self._saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._out, self._err

    def readouterr(self) -> CaptureResult:
        result = CaptureResult(self._out.getvalue(), self._err.getvalue())
        for stream in (self._out, self._err):
            stream.seek(0)
            stream.truncate()
        return result

    def close(self) -> None:
        sys.stdout, sys.stderr = self._saved


class Request:
    """What a fixture can learn about the test requesting it."""

    def __init__(self, case: AOPTestCase, marks: set[str]) -> None:
        self._case = case
        self.marks = marks

    def getfixturevalue(self, name: str) -> Any:
        return self._case.fixture_value(name)


class AOPTestCase(unittest.TestCase):
    """A test case whose test methods take fixtures and parameters by name."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, test in list(vars(cls).items()):
            if not name.startswith("test") or not inspect.isfunction(test):
                continue
            cases = getattr(test, "_parameters", [{}])
            delattr(cls, name)
            for index, parameters in enumerate(cases):
                suffix = f"_{index}_{_label(parameters)}" if len(cases) > 1 else ""
                setattr(cls, name + suffix, _bound(test, parameters))

    def setUp(self) -> None:
        import shared_fixtures  # noqa: F401  registers the shared fixtures

        self._values: dict[str, Any] = {}
        self._stack = contextlib.ExitStack()
        self.addCleanup(self._stack.close)
        test = getattr(self, self._testMethodName)
        self._marks = set(getattr(test, "_marks", set()))
        for name, (_, autouse) in list(_FIXTURES.items()):
            if autouse:
                self.fixture_value(name)

    def fixture_value(self, name: str) -> Any:
        if name in self._values:
            return self._values[name]
        if name == "tmp_path":
            value: Any = Path(tempfile.mkdtemp(prefix="aop-test-")).resolve()
            self._stack.callback(_remove_tree, value)
        elif name == "monkeypatch":
            value = MonkeyPatch()
            self._stack.callback(value.undo)
        elif name == "capsys":
            value = Capture()
            self._stack.callback(value.close)
        elif name == "request":
            value = Request(self, self._marks)
        else:
            try:
                function, _ = _FIXTURES[name]
            except KeyError as error:
                raise LookupError(f"no fixture named {name!r}") from error
            arguments = {
                parameter: self.fixture_value(parameter)
                for parameter in inspect.signature(function).parameters
            }
            value = function(**arguments)
            if inspect.isgenerator(value):
                generator = value
                value = next(generator)
                self._stack.callback(next, generator, None)
        self._values[name] = value
        return value


def _bound(test: Callable[..., Any], parameters: dict[str, Any]) -> Callable[..., Any]:
    names = list(inspect.signature(test).parameters)[1:]

    @functools.wraps(test)
    def run(self: AOPTestCase) -> Any:
        return test(
            self,
            **{
                name: parameters[name]
                if name in parameters
                else self.fixture_value(name)
                for name in names
            },
        )

    run.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    return run


def _label(parameters: dict[str, Any]) -> str:
    text = "_".join(str(value) for value in parameters.values())
    return re.sub(r"\W+", "_", text).strip("_")[:60]


def _remove_tree(path: Path) -> None:
    def writable(function: Callable[..., Any], target: str, _: Any) -> None:
        with contextlib.suppress(OSError):
            os.chmod(os.path.dirname(target), stat.S_IRWXU)
            os.chmod(target, stat.S_IRWXU)
            function(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=writable)
    else:
        shutil.rmtree(path, onerror=writable)

