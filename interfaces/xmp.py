def init_py(lib, comm):
    fcomm = comm.py2f()
    lib.xmp_init_py(fcomm)

def finalize_py(lib):
    lib.xmp_finalize()
