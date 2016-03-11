README

# How to install
 See "INSTALL.md"

-----
# Implementation Status
 Some features in the language specification are not supported in this release.
 (You can download the specification of XMP from "http://xcalablemp.org")

 See "docs/STATUS/$(version).md"

-----
# How to use
## Compile
    $ xmpcc -O2 test.c -o test
    $ xmpf90 -O2 test.f90 -o test

## Compile (OpenACC)
    $ ompcc -acc -O2 test.c -o test

## Run
 Please use the mpi command on your environment.
    $ mpirun -np 4 test

## Environment Variables
* XMP_ONESIDED_HEAP_SIZE
 **Note that on the K computer this value is not used.**
 This value specifies memory size for coarray. The default size is 16MB.
 To set this value, please execute as follows:

    $ export XMP_ONESIDED_HEAP_SIZE=32M

* XMP_ONESIDED_STRIDE_SIZE
 **Note that on the K computer this value is not used.**
 This value specifies memory size for coarray stride operation.
 The default size is 1MB.
 To set this value, please execute as follows:

    $ export XMP_ONESIDED_STRIDE_SIZE=2M

* XMP_NODE_SIZEn
 This value specifies the extent of the n'th dimension of a non-primary node
 array that is declaread as '*' and not the last one. Note that n is 0-origin.
 For example, when XMP_NODE_SIZE0 and XMP_NODE_SIZE1 are set as follows:

    $ export XMP_NODE_SIZE0=4
    $ export XMP_NODE_SIZE1=4

 and the program is run on 32 nodes in all, the shape of the node array p declared as
 follows is assumed to be 4x4x2.

    !$xmp nodes p(*,*,*)

## Compiler Options
* --openmp
* -omp
 These options enable handling of OpenMP directives.

* -max_assumed_shape=N
 This option specifies the maximum number of assumed-shape array arguments of an XMP/F
 procedure. The default is 16. If the number of them exceeds this value, the result is
 not guaranteed.

* -J dir
 This option adds the directory "dir" to the list of directories to be searched for and put
 module information files by the compiler.

-----
# Usage of BLAS in XMP library
 For example the function xmp_matmul(), BLAS can be used internally to improve performance.
 To do this, "--with-libblas=" should be used in executing ./configure.

* When using "--with-libblas=",
  Indicatied BLAS is used in the XMP library, and the BLAS is added to linking phase in
  xmpcc/xmpf90.

* When not using "--with-libblas=",
** On the K computer
  SSL2BLAMP is used automatically. We recommend this way.

** When using "--enable-SSL2BLAMP" in executing ./configure.
  SSL2BLAMP is used in the XMP library. This option is for FX10 and FX100.

** When using "--enable-intelmkl" in executing ./configure.
  Intel MKL is used in the XMP library. This option is for Intel compiler.

** Other
  Instead of BLAS, functions prepared by the XMP library is used.

-----
# Profiling Options in XMP/C
 Omni XMP compiler supports profiler interfaces of Scalasca and tlog.
 The tlog is included in this package. To use the profiler interface of Scalasca,
 you need to download and compile it from Scalasca website (http://www.scalasca.org).
 To use the scalasca, please set the following environmental variable.

    $ export SCALASCA_HOME = [Scalasca-INSTALL-DIR]

 To specify directives for profiling, please add "profile" to directive.

     #pragma xmp loop on t(i) profile

