import logging
import logging.config
import os
import pathlib
from typing import Any

# -Config for the dataset- #
ROOT = pathlib.Path(__file__).parent.parent
LOG_FOLDER = ROOT / "res/logs"
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_FILE_PATH = LOG_FOLDER / "log_msgs.log"

RES_FOLDER = ROOT / "pyvisim/res"
MODEL_FILES_PATH = RES_FOLDER / "model_files"
LOGGER_FILE = RES_FOLDER / "logging_config.yaml"
LOCAL_LOGGER_FILE = ROOT / "logger_config.yaml.local"

_BOOLEANS = {
    "true": True,
    "yes": True,
    "on": True,
    "false": False,
    "no": False,
    "off": False,
}


def _parse_scalar(token: str) -> Any:
    """Convert a single YAML scalar token into its Python value."""
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    lowered = token.lower()
    if lowered in _BOOLEANS:
        return _BOOLEANS[lowered]
    if lowered in ("null", "~", ""):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment that sits outside of a quoted string."""
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _parse_block(
    lines: list[tuple[int, str]], start: int, indent: int
) -> tuple[dict[str, Any], int]:
    """Parse the mapping at ``indent`` starting at ``lines[start]``.

    Returns the parsed mapping and the index of the first unconsumed line.
    """
    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation in logging config: {content!r}")
        key, separator, rest = content.partition(":")
        if not separator:
            raise ValueError(f"Malformed line in logging config: {content!r}")
        key = key.strip()
        rest = rest.strip()
        if rest:
            mapping[key] = _parse_scalar(rest)
            index += 1
            continue
        # A bare ``key:`` opens a nested block, unless nothing is indented
        # underneath it, in which case the value is empty.
        if index + 1 < len(lines) and lines[index + 1][0] > indent:
            mapping[key], index = _parse_block(lines, index + 1, lines[index + 1][0])
        else:
            mapping[key] = None
            index += 1
    return mapping, index


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    """Parse the subset of YAML used by the logging configuration.

    Supports nested mappings, inline flow sequences, comments and the scalar
    types (``str``, ``int``, ``float``, ``bool``, ``None``) that
    :func:`logging.config.dictConfig` expects. Block sequences, anchors,
    multi-line strings and multiple documents are *not* supported.
    """
    lines: list[tuple[int, str]] = []
    with open(path) as file:
        for raw_line in file:
            content = _strip_comment(raw_line.rstrip())
            if not content.strip():
                continue
            if content.lstrip().startswith("- "):
                raise ValueError("Block sequences are not supported; use [a, b]")
            lines.append((len(content) - len(content.lstrip()), content.strip()))
    if not lines:
        return {}
    config, _ = _parse_block(lines, 0, lines[0][0])
    return config


# - Logging - #
def setup_logging(
    default_path: str | pathlib.Path = LOGGER_FILE,
    local_path: str | pathlib.Path = LOCAL_LOGGER_FILE,
    default_level: int = logging.WARNING,
) -> None:
    """Setup logging configuration.

    ``local_path`` lets a developer keep an uncommitted ``logger_config.yaml.local``
    at the repository root; when present it takes precedence over ``default_path``.
    A missing or unusable local file falls back to the committed configuration,
    and an unusable committed configuration falls back to :func:`logging.basicConfig`.
    """
    candidates = [default_path]
    if pathlib.Path(local_path).is_file():
        candidates.insert(0, local_path)

    for path in candidates:
        try:
            config = load_yaml(path)
            try:
                config["handlers"]["file_handler"]["filename"] = str(LOG_FILE_PATH)
            except Exception as e:
                print(
                    f"Error in Logging Configuration: {e}. Cannot set output path for log file."
                )
            logging.config.dictConfig(config)
            return
        except Exception as e:
            print(f"Error in Logging Configuration ({path}): {e}")

    logging.basicConfig(
        level=default_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
