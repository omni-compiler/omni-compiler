_init_lib = None
def init(libfile, comm):
    import ctypes
    global _init_lib
    fcomm     = comm.py2f()
    _init_lib = ctypes.CDLL(libfile)
    _init_lib.xmp_init_py(fcomm)
    return _init_lib

def finalize():
    _init_lib.xmp_finalize()

class spawn:
    def __init__(self, nodes, libfile, funcname, async=False):
        self._nodes    = nodes
        self._libfile  = libfile
        self._funcname = funcname
        self._isAsync  = async
        self._comm     = None
        self._running  = False
        
    def __del__(self):
        spawn.wait(self)
        
    def run(self, *args):
        from mpi4py import MPI
        import tempfile, os, sys
        
        tmpf = tempfile.NamedTemporaryFile(delete=False, dir="./")
        tmpf.write("from mpi4py import MPI\n")
        tmpf.write("import numpy\n")
        tmpf.write("from ctypes import *\n\n")
        tmpf.write("comm = MPI.Comm.Get_parent()\n")
    
        for (i,a) in enumerate(args):
            argname = "arg" + str(i)
            tmpf.write(argname + " = numpy.zeros(" + str(a.size) + ")\n")
            tmpf.write("comm.Bcast(" + argname + ", root=0)\n")
        
        tmpf.write("lib = CDLL(\"" + self._libfile + "\")\n")
        tmpf.write("lib.xmp_init_py(comm.py2f())\n")

        tmpf.write("lib." + self._funcname + "(")
        for (i,a) in enumerate(args):
            tmpf.write("arg" + str(i) + ".ctypes")
            if(i != len(args)-1):
                tmpf.write(",")
        
        tmpf.write(")\n")
    
        tmpf.write("lib.xmp_finalize()\n")
        tmpf.write("comm.Disconnect()\n")
        tmpf.close()
        self._comm = MPI.COMM_SELF.Spawn(sys.executable, args=[tmpf.name], maxprocs=self._nodes)
        for a in args:
            self._comm.Bcast(a, root=MPI.ROOT)
            
        os.unlink(tmpf.name)
        if self._isAsync == False:
            self._comm.Disconnect()
        else:
            self._running = True
        
    def wait(self):
        if self._isAsync and self._running:
           self._comm.Disconnect()
           self._running = False
