import xmp

lib = xmp.Lib("test.so")
job = lib.spawn(4, "hello_2", ([1,2,3], [4,5,6]))

#lib = XMP.Lib("test.so")
#job = lib.spawn(4, "hello_1", [1,2,3])

#lib = XMP.Lib("test.so")
#job = lib.spawn(4, "hello_0")

print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")


