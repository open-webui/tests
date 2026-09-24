"""Read the checkout's Dockerfile the way docker reads it, for the container build guards.

`read_dockerfile(repo_root)` returns its stages in order, the first holding the ARGs declared
before any FROM. Comment lines are dropped before backslash continuations are joined, as docker
does, so a comment inside a multi-line ENV does not end it. Each instruction keeps its arguments
and the WORKDIR in effect. `shell_commands(run)` splits a RUN body into simple commands,
`served_port(repo_root)` is the port the image's server listens on.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

CONTROL_OPERATORS = {";", "&&", "||", "|", "&", ";;"}
SHELL_KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{", "}"}


@dataclass
class Instruction:
    keyword: str
    arguments: str
    workdir: str

    def words(self) -> list[str]:
        return shlex.split(self.arguments)


@dataclass
class Stage:
    name: str
    base: str
    instructions: list[Instruction] = field(default_factory=list)

    def of(self, keyword: str) -> list[Instruction]:
        return [instruction for instruction in self.instructions if instruction.keyword == keyword]

    def env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for instruction in self.of("ENV"):
            words = instruction.words()
            if words and "=" not in words[0]:
                values[words[0]] = " ".join(words[1:])  # the legacy `ENV KEY value` form
                continue
            values.update(word.split("=", 1) for word in words)
        return values

    def copied_to(self, filename: str) -> list[str]:
        """Where this stage's COPY and ADD instructions put a file named `filename`."""
        destinations = []
        for instruction in self.of("COPY") + self.of("ADD"):
            *sources, destination = [
                word for word in instruction.words() if not word.startswith("--")
            ]
            if not any(PurePosixPath(source).name == filename for source in sources):
                continue
            target = PurePosixPath(instruction.workdir) / destination
            is_directory = destination.endswith("/") or len(sources) > 1
            destinations.append(str(target / filename if is_directory else target))
        return destinations


@dataclass
class Dockerfile:
    stages: list[Stage]

    @property
    def global_args(self) -> Stage:
        return self.stages[0]

    @property
    def build_stages(self) -> list[Stage]:
        return self.stages[1:]

    @property
    def runtime(self) -> Stage:
        return self.stages[-1]


def _logical_lines(text: str) -> list[str]:
    kept = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    joined = re.sub(r"\\[ \t]*\n", " ", "\n".join(kept))
    return [line.strip() for line in joined.splitlines() if line.strip()]


def parse_dockerfile(text: str) -> Dockerfile:
    stages = [Stage(name="<global>", base="")]
    workdir = "/"
    for line in _logical_lines(text):
        keyword, _, arguments = line.partition(" ")
        keyword = keyword.upper()
        if keyword == "FROM":
            words = [word for word in arguments.split() if not word.startswith("--")]
            name = words[2] if len(words) > 2 and words[1].upper() == "AS" else words[0]
            stages.append(Stage(name=name, base=words[0]))
            workdir = "/"
            continue
        if keyword == "WORKDIR":
            workdir = str(PurePosixPath(workdir) / arguments.strip())
        stages[-1].instructions.append(Instruction(keyword, arguments.strip(), workdir))
    assert len(stages) > 1, "the Dockerfile has no FROM instruction"
    return Dockerfile(stages)


def read_dockerfile(repo_root: Path) -> Dockerfile:
    path = repo_root / "Dockerfile"
    assert path.is_file(), f"no Dockerfile at {path}; retarget the container build guards"
    return parse_dockerfile(path.read_text(encoding="utf-8"))


def shell_commands(script: str) -> list[list[str]]:
    """The simple commands of a RUN body, split at control operators, leading keywords dropped."""
    lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    commands: list[list[str]] = [[]]
    for token in lexer:
        if token in CONTROL_OPERATORS:
            commands.append([])
        elif commands[-1] or token not in SHELL_KEYWORDS:
            commands[-1].append(token)
    return [command for command in commands if command]


def run_commands(stage: Stage) -> list[list[str]]:
    """Every simple command the stage's RUN instructions execute, RUN flags excluded."""
    commands = []
    for run in stage.of("RUN"):
        script = re.sub(r"^(--\S+\s+)*", "", run.arguments)
        commands.extend(shell_commands(script))
    return commands


def served_port(repo_root: Path) -> str:
    """The port start.sh serves on inside the image: the runtime stage's `ENV PORT`."""
    port = read_dockerfile(repo_root).runtime.env().get("PORT")
    assert port, (
        "the Dockerfile's runtime stage no longer sets ENV PORT; retarget served_port at where "
        "the image now sets the port start.sh listens on"
    )
    return port
