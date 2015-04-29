# Needed softwares
 * Lex, Yacc
 * C and Fortran Compilers (supports C99 and Fortran 90)
 * C++
 * Java Compiler
 * MPI Implementation (supports MPI-2)
 * libxml2
 * make

## On Debian GNU/Linux and Ubuntu
 The following packages may be needed to be installed previously:

 * flex gcc gfortran g++ openjdk-7-jdk libopenmpi-dev openmpi-bin libxml2-dev byacc make perl

## On Red Hat and CentOS
 The following packages may be needed to be installed previously:

 * flex gcc gfortran gcc-c++ java-1.7.0-openjdk-devel openmpi-devel libxml2-devel byacc make perl

## For local-view operations (coarray, post/wait, lock/unlock)
 * On the K computer or FX10, the Omni compiler automatically uses Fujitsu RDMA.
 * On the other system, please install GASNet (http://gasnet.lbl.gov) before installing the Omni compiler.

## For Omni OpenACC compiler
 * Need to install CUDA (https://developer.nvidia.com/cuda-zone).

# Install Step
## Configure
### On a general linux cluster
    $ ./configure --prefix=[INSTALLATION PATH]
         or
    $ ./configure CC=gcc FC=gfortran   // PGI compiler

 If you want to use Coarray functions
    $ ./configure --with-gasnet=[GASNet INSTALLATION PATH] --with-gasnet-conduit=[GASNet-Conduit]

    The "GASNet-Conduit" is a method how GASnet uses an interconnect.

    If you omit "--with-gasnet-conduit=[GASNet-Conduit]",
    the Omni compiler automatically selects an appropriate conduit.

    If you specify "--with-gasnet-conduit=mpi", the execute file can execute on the most clusters.
    If a running system is equipped with InfiniBand, "--with-gasnet-conduit=ibv" is the best selection.
    For information of other conduits, please see the GASNet website (http://gasnet.lbl.gov).

 If you want to use OpenACC compiler
    $ ./configure --enable-openacc --with-cuda=[CUDA INSTALLATION PATH]
    
    If you want to compile for Kepler or newer GPU, you should specify '--with-gpu-cflags="-arch=sm_XX -O3"'
    (XX is compute capability version of the GPU).

### On the K computer
    $ ./configure --target=Kcomputer-linux-gnu --prefix=[INSTALLATION PATH]

### On FX10
    $ ./configure --target=FX10-linux-gnu --prefix=[INSTALLATION PATH]

### On Cray machines
    $ ./configure --target=Cray-linux-gnu --prefix=[INSTALLATION PATH]

### On NEC SX machines
    $ ./configure --target=sx-nec-superux --prefix=[INSTALLATION PATH]

### On IBM BlueGene/Q
    We recommend to install openJDK for AIX (e.g. openjdk1.7.0-ppc-aix-port-linux-ppc64-b**.tar.bz2,
    from http://cr.openjdk.java.net/~simonis/ppc-aix-port/).
    $ ./configure --target=powerpc-ibm-cnk --prefix=[INSTALLATION PATH]

### On HITACHI SR16000 machines
   $ ./configure --target=powerpc-hitachi-aix --prefix=[INSTALLATION PATH]

## Build
    $ make; make install

## Set PATH
    $ export PATH=[INSTALLATION PATH]/bin:$PATH

## Test (Optional)
    $ make tests
    $ make run-tests

 If you want to clean binaries of tests,
    $ make clean-tests

# Note
 In some reasons you failed to build the XMP, especially in the configuration stage,
 issuing autogen.sh on the top directory would be a solution.
 And if the build still failed even after issuing the autogen.sh,
 you should update your autotools (autoconf/automake/libtools) to the latest ones.
