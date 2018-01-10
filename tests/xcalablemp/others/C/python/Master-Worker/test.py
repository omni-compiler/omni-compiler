import xmp

#prog = xmp.Program("test.so", "hello_2")
#job   = prog.spawn(4, ([1,2,3], [4,5,6]))

#prog = xmp.Program("test.so", "hello_1")
#job  = prog1.spawn(4, [1,2,3])

prog = xmp.Program("test.so", "hello_0")
job   = prog.spawn(4)


print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")


