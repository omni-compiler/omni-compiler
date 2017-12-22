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

class xmp:
    def __init__(self, nodes, libfile, funcname):
        self._nodes    = nodes
        self._libfile  = libfile
        self._funcname = funcname
        
    def spawn(self, *args):
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
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=[tmpf.name], maxprocs=self._nodes)
        for a in args:
            comm.Bcast(a, root=MPI.ROOT)
        
        comm.Disconnect()
        os.unlink(tmpf.name)
   
