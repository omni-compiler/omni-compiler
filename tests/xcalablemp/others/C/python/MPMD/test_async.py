import xmp
import numpy

arg1 = numpy.array([1,2,3])
arg2 = numpy.array([4,5,6])
hello = xmp.spawn(4, "test.so", "hello", async=True)
hello.run(arg1, arg2)
print "TEST ASYNC"
hello.wait()

        
