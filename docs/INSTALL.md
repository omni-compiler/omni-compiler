HOW TO INSTALL

# Needed softwares
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
    $ ./configure --prefix=[INSTALL-DIR]
         or
    $ ./configure --with-backend-cc=mpicc --prefix=[INSTALL-DIR]

 If you want to use Coarray functions
    $ ./configure --with-gasnetDir=[GASNet-INSTALL-DIR] --with-gasnet-conduit=[GASNet-Conduit] --prefix=[INSTALL-DIR]

### On the K computer
    $ ./configure --target=Kcomputer-linux-gnu --enable-fjrdma --prefix=[INSTALL-DIR]

### On Cray machines
    $ ./configure --target=Cray-linux-gnu --prefix=[INSTALL-DIR]

### On SX machines
    $ ./configure --target=sx --prefix=[INSTALL-DIR]

## Build
    $ make; make install

## Set PATH
    $ export PATH=[INSTALL-DIR]/bin:$PATH

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

