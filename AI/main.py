
from segmentation import *

if __name__ == "__main__":
    dic = {}
    dic["./Utils/coleoi.jpg"] = []
    dictReturn = segmentDict(dic)
    print(dictReturn)
