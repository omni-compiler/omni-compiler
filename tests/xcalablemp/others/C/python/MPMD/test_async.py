import xmp

prog1 = xmp.Program("test.so", "hello")
prog1.spawn(4, [1,2,3], [4,5,6], async=True)
print("HELLO ASYNC")
prog1.wait()


