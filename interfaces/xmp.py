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
        self._comm    = None
        self._running = False
        self._isAsync = False
        
#    def __del__(self):
#        Program.wait(self)
                
    def spawn(self, nodes, *args, async=False):
        from mpi4py import MPI
        import tempfile, os, sys, numpy
        
        self._isAsync = async
        tmpf = tempfile.NamedTemporaryFile(delete=False, dir="./")
        tmpf.write(b"from mpi4py import MPI\n")
        tmpf.write(b"import numpy\n")
        tmpf.write(b"from ctypes import *\n")
        tmpf.write(b"comm = MPI.Comm.Get_parent()\n")

        for (i,a) in enumerate(args):
            tmp_a = numpy.array(a)
            argname = "arg" + str(i)
            tmpf.write(argname.encode() + b" = numpy.zeros(" + str(tmp_a.size).encode() + b")\n")
            tmpf.write(b"comm.Bcast(" + argname.encode() + b", root=0)\n")
 
        tmpf.write(b"lib = CDLL(\"" + self.so.encode() + b"\")\n")
        tmpf.write(b"lib.xmp_init_py(comm.py2f())\n")

        tmpf.write(b"lib." + self.name.encode() + b"(")
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
            
        if self._isAsync == False:
            self._comm.Disconnect()
        else:
            self._running = True
            
        os.unlink(tmpf.name)
        
    def wait(self):
        if self._isAsync and self._running:
            self._comm.Disconnect()
            self._running = False
