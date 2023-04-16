#1 parser
from utils.parser import parser_args
#2 평균, 분산
from utils.tools import CIFAR_MEAN, CIFAR_STD
#3 데이터로더
from utils.getModules import getDataLoader
def main():
    #1 parser_args
    args=parser_args()
    #
    train_loader, test_loader = getDataLoader(args)