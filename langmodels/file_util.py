import logging
import os
from typing import Generator


logger = logging.getLogger(__name__)


def check_path_writable(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)
    elif not os.access(path, os.W_OK | os.X_OK):
        raise FileNotFoundError(path)


def check_path_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def get_all_files(path: str) -> Generator[str, None, None]:
    if os.path.isfile(path):
        yield path
    else:
        for root, dirs, files in os.walk(path, followlinks=True):
            for file in files:
                if file.endswith('.java'):
                    yield os.path.join(root, file)


def read_file_with_encoding(file_path: str, encoding: str) -> str:
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def read_file_contents(file_path: str) -> str:
    try:
        return read_file_with_encoding(file_path, 'utf-8')
    except UnicodeDecodeError:
        try:
            return read_file_with_encoding(file_path, 'ISO-8859-1')
        except UnicodeDecodeError:
            logger.error(f"Unicode decode error in file: {file_path}")


def get_file_total_lines(file):
    with open(file, 'r') as f:
        return len([i for i in f])


def get_file_extension(file: str) -> str:
    """
    >>> get_file_extension('new_file.java')
    'java'
    """
    return os.path.splitext(file)[1][1:]