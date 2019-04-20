import xmp
import numpy

arg1 = numpy.array([1,2,3])
arg2 = numpy.array([4,5,6])
lib  = xmp.Lib("test.so")
job  = lib.spawn(4, "hello_2", (arg1, arg2))

print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")


