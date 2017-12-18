import xmp
import numpy
from ctypes import *

n1 = numpy.array([1,2,3])
xmp.spawn("test.so", 4, "hello", n1)
        
