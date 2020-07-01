import xmp

lib = xmp.Lib("test.so")
job = lib.spawn(4, "hello_2", ([1,2,3], [3,4,5]), async=True)
print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")
job.wait()
print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")
