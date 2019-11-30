import logging
import os

from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


def check_path_writable(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.access(path, os.W_OK | os.X_OK):
        raise FileNotFoundError(path)


def check_path_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def get_all_files(path: str, extension: Optional[str] = 'java') -> Generator[Path, None, None]:
    if os.path.isfile(path):
        yield Path(path)
    else:
        for root, dirs, files in os.walk(path, followlinks=True):
            for file in files:
                if not extension or file.endswith(f'.{extension}'):
                    yield Path(os.path.join(root, file))


def read_file_with_encoding(file: Path, encoding: str) -> str:
    with file.open('r', encoding=encoding) as f:
        return f.read()


def read_file_contents(file: Path) -> str:
    try:
        return read_file_with_encoding(file, 'utf-8')
    except UnicodeDecodeError:
        try:
            return read_file_with_encoding(file, 'ISO-8859-1')
        except UnicodeDecodeError:
            logger.error(f"Unicode decode error in file: {file}")


def get_file_total_lines(file):
    with open(file, 'r') as f:
        return len([i for i in f])


def get_file_extension(file: str) -> str:
    """
    >>> get_file_extension('new_file.java')
    'java'
    """
    return os.path.splitext(file)[1][1:]