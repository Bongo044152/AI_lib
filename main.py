import logging
from utils.my_logger import init_logger
import model
import pytest
from model.DeepSeekModel import DeepSeekController, DeepSeekOption
from packaging import version
import pydantic

assert version.parse(pydantic.VERSION) >= version.parse("2.0.0"), "require: pydantic.VERSION >= v2"

def main():
    init_logger()
    deepseek = model.DeepSeekModel(model="gpt-40")

if __name__ == '__main__':
    main()