import xmp

prog1 = xmp.Program("test.so", "hello")
job   = prog1.run(4, ([1,2,3], [3,4,5]), async=True)
print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")
job.wait()
print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")
