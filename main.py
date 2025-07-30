import logging
from utils.my_logger import init_logger
import model

def main():
    init_logger()
    claud = model.ClaudeController()


if __name__ == '__main__':
    main()