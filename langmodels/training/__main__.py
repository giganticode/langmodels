import sys

from langmodels.training.cli import run


def main():
    run(sys.argv[1:])


if __name__ == '__main__':
    main()
