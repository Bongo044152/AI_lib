import logging
from utils.my_logger import init_logger
import model

from packaging import version
import pydantic

assert version.parse(pydantic.VERSION) >= version.parse("2.0.0"), "require: pydantic.VERSION >= v2"

def main():
    init_logger()
    claud = model.ClaudeController()


if __name__ == '__main__':
    main()