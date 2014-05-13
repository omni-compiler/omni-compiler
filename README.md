# Getting Started
## Needed softwares
 * Lex, Yacc
 * C and Fortran Compilers (supports C99 and Fortran 90)
 * C++
 * Java Compiler
 * Apache Ant (1.8.1 or later)
 * MPI Implementation (supports MPI-2)
 * libxml2
 * make

## On Debian GNU/Linux and Ubuntu
 The following packages may be needed to be installed previously:

* flex gcc gfortran g++ openjdk-7-jdk ant libopenmpi-dev openmpi-bin libxml2-dev byacc make

## On Red Hat and CentOS
 The following packages may be needed to be installed previously:

* flex gcc gfortran gcc-c++ java-1.7.0-openjdk-devel ant openmpi-devel libxml2-devel byacc make

## Usage of coarray functions
 * Need to install GASNet (http://gasnet.lbl.gov) except for the K computer.
 * On the K computer, attach "--enable-fjrdma" option when executing ./configure.


# Install Step
## Configure
### On a general linux cluster
 $ ./configure --prefix=[INSTALL DIR]
         or
 $ ./configure --with-backend-cc=mpicc --prefix=[INSTALL DIR]

 If you want to use Coarray functions
 $ ./configure --with-gasnetDir=[GASNet INSTALL DIR] --with-gasnet-conduit=[GASNet Conduit] --prefix=[INSTALL DIR]

### On the K computer
 $ ./configure --target=Kcomputer-linux-gnu --enable-fjrdma --prefix=[INSTALL DIR]

### On a Cray machine
 $ ./configure --target=Cray-linux-gnu --prefix=[INSTALL DIR]

## Build
 $ make; make install

## Set PATH
 e.g. ) export PATH=[INSTALL DIR]/bin:$PATH

## Test (Optional)
 $ make tests
 $ make run-tests

 If you want to clean binaries of tests,
 $ make clean-tests 

# How to use
## Compile
 e.g.) $ xmpcc -O2 test.c -o test
       $ xmpf90 -O2 test.f90 -o test
   
## Run
 Please use the command for runing mpi program on your environment.
 e.g.) $ mpirun -np 4 test

## Environment Variable
 - XMP_COARRAY_HEAP_SIZE : 
   The heap memory size for using coarray. The default size is 256MB.
   If you want to change this size, please set a value by the Mega Byte.

   e.g.) export XMP_COARRAY_HEAP_SIZE=128

# Profiling Options in XMP/C
 XMP supports profiler interfaces of Scalasca and tlog.
 The tlog is included in XMP.
 If you want to use the profiler interface of Scalasca,
 you need to download and compile it from Scalasca website.
 Moreover you need to set an environmental variable as below,
 $ export SCALASCA_HOME = [Scalasca INSTALL DIR]

 If you want to specify directives for profiling,
 please add "profile" to directive.
 e.g.) #pragma xmp loop on t(i) profile

 When you want to compile XMP source code with profiling,
 please add some of below options.
  -profile         Emit XMP directive profiling code only for specified directives
  -allprofile      Emit XMP directive profiling code for all directives
  -with-scalasca   Emit Scalasca instrumentation
  -with-tlog       Emit tlog insturumentation


# Implementation Status
 Some features in the language specification are not supported in this release.
 (You can download the specification of XMP from "http://www.xcalablemp.org".)

 Please show http://www.hpcs.cs.tsukuba.ac.jp/omni-compiler/xcalablemp/download.html

# Note
 In some reasons you failed to build the XMP, especially in the configuration stage, 
 issuing autogen.sh on the top directory would be a solution. 
 And if the build still failed even after issuing the autogen.sh, 
 you should update your autotools (autoconf/automake/libtools) to the latest ones.

