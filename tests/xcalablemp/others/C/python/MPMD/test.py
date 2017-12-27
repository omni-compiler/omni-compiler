import xmp                                                    
import numpy

prog1 = xmp.Program("test.so", "hello")
job   = prog1.spawn(4, numpy.array([1,2,3]), numpy.array([1,2,3]))

print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")

