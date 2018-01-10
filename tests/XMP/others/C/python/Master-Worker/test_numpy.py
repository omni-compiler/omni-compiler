import xmp
import numpy

arg1 = numpy.array([1,2,3])
arg2 = numpy.array([4,5,6])
prog = xmp.Program("test.so", "hello_2")
job  = prog.spawn(4, (arg1, arg2))

print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")


