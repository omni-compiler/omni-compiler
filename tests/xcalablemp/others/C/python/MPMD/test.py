import xmp

prog1 = xmp.Program("test.so", "hello")
job   = prog1.run(4, ([1,2,3], [3,4,5]))
#job   = prog1.run(4, [1,2,3])
#job2   = prog1.run(4)
#print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")

