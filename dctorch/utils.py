import torch
import sys,torch,logging
from pynvml import *

logging.info("torch: {0} ".format(torch.__version__))
class gpu(object):
    def __init__(self):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        self.__isgpu = torch.cuda.is_available()
        assert self.__isgpu, "未找到CUDA，请检查"
        nvmlInit()
