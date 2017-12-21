from mpi4py import MPI
import xmp

lib = xmp.init('test.so', MPI.COMM_WORLD)
lib.hello()
xmp.finalize()
