from boundinnerclass import BoundInnerClass
import time

_so = None
def init(so, comm):
    import ctypes
    global _so
    fcomm = comm.py2f()
    _so   = ctypes.CDLL(so)
    _so.xmp_init_py(fcomm)
    return _so

def finalize():
    _so.xmp_finalize()

class Program:
    def __init__(self, so="", name=""):
        self.so       = so
        self.name     = name
            
    @BoundInnerClass
    class run:
        def __init__(self, outer, nodes, *args, async=False):
            from mpi4py import MPI
            import tempfile, os, sys, numpy

            self.start_time = time.time()
            self.time       = 0
            self._comm      = None
            self._isAsync   = async

            has_args = args != ()
            if has_args:
                if isinstance(args[0], tuple):
                    tuple_args = args[0]
                else:
                    tuple_args = args
            
            f = tempfile.NamedTemporaryFile(delete=False, dir="./")
            f.write(b"from ctypes import *\n")
            f.write(b"from mpi4py import MPI\n")
            f.write(b"comm = MPI.Comm.Get_parent()\n")
            
            if has_args:
                f.write(b"import numpy\n")

                for (i,a) in enumerate(tuple_args):
                    tmp_a = numpy.array(a)
                    argname = "arg" + str(i)
                    f.write(argname.encode() + b" = numpy.zeros(" + str(tmp_a.size).encode() + b")\n")
                    f.write(b"comm.Bcast(" + argname.encode() + b", root=0)\n")

            f.write(b"lib = CDLL(\"" + outer.so.encode() + b"\")\n")
            f.write(b"lib.xmp_init_py(comm.py2f())\n")
            f.write(b"lib." + outer.name.encode() + b"(")

            if has_args:
                for (i,a) in enumerate(tuple_args):
                    f.write(b"arg" + str(i).encode() + b".ctypes")
                    if(i != len(tuple_args)-1):
                        f.write(b",")
        
            f.write(b")\n")
    
            f.write(b"lib.xmp_finalize()\n")
            f.write(b"comm.Disconnect()\n")
            f.close()
            
            self._comm = MPI.COMM_SELF.Spawn(sys.executable, args=[f.name], maxprocs=nodes)
            if has_args:
                for a in tuple_args:
                    tmp_a = numpy.array(a)
                    self._comm.Bcast(tmp_a, root=MPI.ROOT)

            os.unlink(f.name)
            
            if self._isAsync == False:
                self._comm.Disconnect()
                self._running = False
            else:
                self._running = True

            self.time = time.time() - self.start_time
            
        def __del__(self):
            self.wait()
        
        def wait(self):
            if self._isAsync and self._running:
                self._comm.Disconnect()
                self._running = False
                self.time     = time.time() - self.start_time
                 
        def elapsed_time(self):
            return self.time

