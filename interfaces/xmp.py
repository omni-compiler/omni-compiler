_init_lib = None
_libname = ""
_nodes   = 0

def init_py(lib, comm):
    global _init_lib
    fcomm   = comm.py2f()
    _init_lib = lib
    lib.xmp_init_py(fcomm)

def finalize_py():
    _init_lib.xmp_finalize()

def spawn(libname, nodes):
    global _libname, _nodes
    _libname = libname
    _nodes   = nodes
    
def run(funcname, *args):
    from mpi4py import MPI
    import tempfile,os,sys

    tmpf = tempfile.NamedTemporaryFile(delete=False, dir="./")
    tmpf.write("from mpi4py import MPI\n")
    tmpf.write("import numpy\n")
    tmpf.write("from ctypes import *\n\n")
    tmpf.write("comm = MPI.Comm.Get_parent()\n")
    
    for (i,a) in enumerate(args):
        argname = "arg" + str(i)
        tmpf.write(argname + " = numpy.zeros(" + str(a.size) + ")\n")
        tmpf.write("comm.Bcast(" + argname + ", root=0)\n")
        
    tmpf.write("lib = CDLL(\"" + _libname + "\")\n")
    tmpf.write("lib.xmp_init_py(comm.py2f())\n")

    tmpf.write("lib." + funcname + "(")
    for (i,a) in enumerate(args):
        tmpf.write("arg" + str(i) + ".ctypes")
        if(i != len(args)-1):
            tmpf.write(",")
        
    tmpf.write(")\n")
    
    tmpf.write("lib.xmp_finalize()\n")
    tmpf.write("comm.Disconnect()\n")
    tmpf.close()
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=[tmpf.name], maxprocs=_nodes)
    for a in args:
        comm.Bcast(a, root=MPI.ROOT)
        
    comm.Disconnect()
    os.unlink(tmpf.name)
   
