#1 parser
from utils.parser import parser_args

#3 데이터로더
from utils.getModules import getDataLoader
def main():
    #1 parser_args
    args=parser_args()
    #3 데이터로더
    train_loader, test_loader = getDataLoader(args)