import time, ctypes, tempfile, os, sys, numpy
from mpi4py import MPI

class Lib:
    def __init__(self, so=""):
        self.so = so
            
    def spawn(self, name, nodes, args=(), async=False):
        job = Job_spawn(name, nodes, self.so, args, async)
        job.run()
        
        return job

    def call(self, comm, name, args=()):
        job = Job_call(comm, name, self.so, args)
        job.run()

        return job
        
class Job_spawn:
    def __init__(self, nodes, name, so, args=(), async=False):
        self.nodes    = nodes
        self.name     = name
        self.so       = so
        self.args     = args if isinstance(args, tuple) else (args,)
        self.async    = async
        self._comm    = None
        self._running = False
        self._time    = 0
        
    def run(self):
        self._start_time = time.time()

        f = tempfile.NamedTemporaryFile(delete=False, dir="./")
        f.write(b"from ctypes import *\n")
        f.write(b"from mpi4py import MPI\n")
        f.write(b"comm = MPI.Comm.Get_parent()\n")
        if self.args != ():
            f.write(b"import numpy\n")

        for (i,a) in enumerate(self.args):
            tmp_a = a if isinstance(a, numpy.ndarray) else numpy.array(a)
            argname = "arg" + str(i)
            f.write(argname.encode() + b" = numpy.zeros(" + str(tmp_a.size).encode() + b")\n")
            f.write(b"comm.Bcast(" + argname.encode() + b", root=0)\n")

        f.write(b"lib = CDLL(\"" + self.so.encode() + b"\")\n")
        f.write(b"lib.xmp_init_py(comm.py2f())\n")
        f.write(b"lib." + self.name.encode() + b"(")

        for (i,a) in enumerate(self.args):
            f.write(b"arg" + str(i).encode() + b".ctypes")
            if(i != len(self.args)-1):
                f.write(b",")
        f.write(b")\n")
        
        f.write(b"lib.xmp_finalize()\n")
        f.write(b"comm.Disconnect()\n")
        f.close()
            
        self._comm = MPI.COMM_SELF.Spawn(sys.executable, args=[f.name], maxprocs=self.nodes)
        for a in self.args:
            tmp_a = numpy.array(a)
            self._comm.Bcast(tmp_a, root=MPI.ROOT)

        os.unlink(f.name)
            
        if self.async:
            self._running = True
        else:
            self._comm.Disconnect()
            self._running = False

        self._time = time.time() - self._start_time
            
    def __del__(self):
        self.wait()
        
    def wait(self):
        if self.async and self._running:
            self._comm.Disconnect()
            self._running = False
            self._time    = time.time() - self._start_time
                 
    def elapsed_time(self):
        return self._time

class Job_call:
    def __init__(self, comm, name, so, args=()):
        self.comm  = comm.py2f()
        self.lib   = ctypes.CDLL(so)
        self.name  = name
        self.args  = args if isinstance(args, tuple) else (args,)
        self._time = 0

    def run(self):
        self._start_time = time.time()

        self.lib.xmp_init_py(self.comm)
        command = "self.lib." + self.name + "("
        tmp_args = []
        for (i,a) in enumerate(self.args):
            tmp_args.append(a if isinstance(a, numpy.ndarray) else numpy.array(a))
            command += "tmp_args[" + str(i) + "].ctypes,"

        if (command[-1] == ","):
            length = len(command)
            command = command[:length-1]
        
        command += ")"
        eval(command)

        self.lib.xmp_finalize()
        self._time = time.time() - self._start_time
        
    def elapsed_time(self):
        return self._time
