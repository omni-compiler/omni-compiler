from mpi4py import MPI
import xmp

prog = xmp.Program("test.so", "hello")
comm = MPI.COMM_WORLD

job  = prog.call(comm, ([1,2,3], [4,5,6]))
#job  = prog.call(comm,[1,2,3])
#job  = prog.call(comm)
comm.Barrier()
if comm.Get_rank() == 0:
    print ("elapsed_time:{0}".format(job.elapsed_time()) + "[sec]")
