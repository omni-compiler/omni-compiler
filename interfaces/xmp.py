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
    class spawn:        
        def __init__(self, outer, nodes, *args, async=False):
            from mpi4py import MPI
            import tempfile, os, sys, numpy

            self.start_time = time.time()
            self.time       = 0
            self._comm      = None
            self._isAsync   = async

            tmpf = tempfile.NamedTemporaryFile(delete=False, dir="./")
            tmpf.write(b"import numpy\n")
            tmpf.write(b"from mpi4py import MPI\n")
            tmpf.write(b"from ctypes import *\n")
            tmpf.write(b"comm = MPI.Comm.Get_parent()\n")

            for (i,a) in enumerate(args):
                tmp_a = numpy.array(a)
                argname = "arg" + str(i)
                tmpf.write(argname.encode() + b" = numpy.zeros(" + str(tmp_a.size).encode() + b")\n")
                tmpf.write(b"comm.Bcast(" + argname.encode() + b", root=0)\n")
                
            tmpf.write(b"lib = CDLL(\"" + outer.so.encode() + b"\")\n")
            tmpf.write(b"lib.xmp_init_py(comm.py2f())\n")
            tmpf.write(b"lib." + outer.name.encode() + b"(")

            for (i,a) in enumerate(args):
                tmpf.write(b"arg" + str(i).encode() + b".ctypes")
                if(i != len(args)-1):
                    tmpf.write(b",")
        
            tmpf.write(b")\n")
    
            tmpf.write(b"lib.xmp_finalize()\n")
            tmpf.write(b"comm.Disconnect()\n")
            tmpf.close()
            
            self._comm = MPI.COMM_SELF.Spawn(sys.executable, args=[tmpf.name], maxprocs=nodes)
            for a in args:
                tmp_a = numpy.array(a)
                self._comm.Bcast(tmp_a, root=MPI.ROOT)

            os.unlink(tmpf.name)
            
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

