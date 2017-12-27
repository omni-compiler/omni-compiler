import xmp

prog1 = xmp.Program("test.so", "hello")
arg1 = [1,2,3]
arg2 = [7,8,9]
prog1.spawn(4, arg1, arg2)


