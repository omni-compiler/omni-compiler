from xmp import *
import numpy

arg1 = numpy.array([1,2,3])
arg2 = numpy.array([4,5,6])
hello = xmp("test.so",2, "hello")
hello.spawn(arg1, arg2)

        
