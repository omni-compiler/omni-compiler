from mpi4py import MPI
import xmp

lib = xmp.Lib("test.so")
comm = MPI.COMM_WORLD

job = lib.call(comm, "hello", ([1,2,3], [4,5,6]))
#job = lib.call(comm,[1,2,3])
#job = lib.call(comm)
comm.Barrier()
if comm.Get_rank() == 0:
    print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")
