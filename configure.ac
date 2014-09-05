#    This file is an input file used by the GNU "autoconf" program to
#    generate the file "configure", which is run during Omni installation
#    to configure the system for the local environment.

AC_PREREQ(2.60)
LT_PREREQ(2.4)

# --------------------------------------------------------------------
# Let the initiation begin....

AC_INIT([Omni Compiler], [0.8.0], [h-murai@riken.jp], [omni])

AC_CONFIG_AUX_DIR([./buildutils])
AC_CONFIG_MACRO_DIR([./buildutils])

AC_CANONICAL_BUILD
AC_CANONICAL_HOST
AC_CANONICAL_TARGET

AC_CONFIG_SRCDIR([include/exc_platform.h])
AC_CONFIG_HEADER([include/config.h Driver/atool/src/include/nata/nata_config.h])

SHELL=`which bash` > /dev/null 2>&1
if test $? -ne 0 -o -z "${SHELL}"; then
   AC_MSG_ERROR([To run libtool you need a bash, not /bin/sh.])
   exit 1
fi
export SHELL
AC_SUBST(SHELL)

LT_INIT
AC_SUBST(LIBTOOL)
AC_SUBST(LIBTOOL_DEPS)

AM_INIT_AUTOMAKE([no-define foreign])
AM_MAINTAINER_MODE([enable])

TOPDIR=${ac_pwd}
AC_SUBST(TOPDIR)

# --------------------------------------------------------------------
# check project dirs
AC_MSG_CHECKING(check XcodeML-Common project exists)

if test -e "XcodeML-Common/Makefile.am"; then
    XCODEML_COMMON_DIR=XcodeML-Common
    AC_CONFIG_FILES($XCODEML_COMMON_DIR/Makefile)
    AC_CONFIG_FILES($XCODEML_COMMON_DIR/ant.properties)
    AC_MSG_RESULT(yes)
else
    XCODEML_COMMON_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(XCODEML_COMMON_DIR)

#----
AC_MSG_CHECKING(check C-FrontEnd project exists)

if test -e "C-FrontEnd/src/Makefile.am"; then
    C_FRONTEND_DIR=C-FrontEnd/src
    AC_CONFIG_FILES($C_FRONTEND_DIR/Makefile)
    AC_MSG_RESULT(yes)
else
    C_FRONTEND_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(C_FRONTEND_DIR)

#----
AC_MSG_CHECKING(check C-BackEnd project exists)

if test -e "C-BackEnd/Makefile.am" -a -e "C-BackEnd/ant.properties.in"; then
    C_BACKEND_DIR=C-BackEnd
    AC_CONFIG_FILES($C_BACKEND_DIR/Makefile)
    AC_CONFIG_FILES($C_BACKEND_DIR/ant.properties)
    AC_CONFIG_FILES($C_BACKEND_DIR/bin/C_Back)
    AC_MSG_RESULT(yes)
else
    C_BACKEND_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(C_BACKEND_DIR)

#----
AC_MSG_CHECKING(check F-FrontEnd project exists)

if test -e "F-FrontEnd/src/Makefile.am"; then
    F_FRONTEND_DIR=F-FrontEnd/src
    AC_CONFIG_FILES($F_FRONTEND_DIR/Makefile)
    AC_MSG_RESULT(yes)
else
    F_FRONTEND_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(F_FRONTEND_DIR)

#----
AC_MSG_CHECKING(check F-BackEnd project exists)

if test -e "F-BackEnd/Makefile.am" -a -e "F-BackEnd/ant.properties.in"; then
    F_BACKEND_DIR=F-BackEnd
    AC_CONFIG_FILES($F_BACKEND_DIR/Makefile)
    AC_CONFIG_FILES($F_BACKEND_DIR/ant.properties)
    AC_CONFIG_FILES($F_BACKEND_DIR/bin/F_Back)
    AC_CONFIG_FILES($F_BACKEND_DIR/bin/F_Trans)
    AC_CONFIG_FILES($F_BACKEND_DIR/bin/F_Decomp)
    AC_MSG_RESULT(yes)
else
    F_BACKEND_DIR=''
    AC_MSG_RESULT(no)
fi

if test "x${C_FRONTEND_DIR}" != "x" -a "x${C_BACKEND_DIR}" != "x"; then
	enableC2C=yes
else
	enableC2C=no
fi

if test "x${F_FRONTEND_DIR}" != "x" -a "x${F_BACKEND_DIR}" != "x"; then
	enableF2F=yes
else
	enableF2F=no
fi

AC_SUBST(F_BACKEND_DIR)

#----
AC_MSG_CHECKING(check XcodeML-Exc-Tools project exists)

if test -e "XcodeML-Exc-Tools/Makefile.am" -a -e "XcodeML-Exc-Tools/ant.properties.in"; then
    XCODEML_EXC_TOOLS_DIR=XcodeML-Exc-Tools
    AC_CONFIG_FILES($XCODEML_EXC_TOOLS_DIR/Makefile)
    AC_CONFIG_FILES($XCODEML_EXC_TOOLS_DIR/ant.properties)
    AC_CONFIG_FILES($XCODEML_EXC_TOOLS_DIR/src/exc/util/MachineDepConst.java)
    AC_MSG_RESULT(yes)
else
    XCODEML_EXC_TOOLS_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(XCODEML_EXC_TOOLS_DIR)

#----
AC_MSG_CHECKING(check Driver project exists)

if test -e "Driver/src/Makefile.am"; then
    OMDRIVER_DIR=Driver/src
    AC_CONFIG_FILES($OMDRIVER_DIR/Makefile)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/ompp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omnative)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omlinker)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/oml2x)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omx2x)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omx2l)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/java.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/pp.conf.openmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/l2x.conf.openmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/x2x.conf.openmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/native.conf.openmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/linker.conf.openmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omc2c)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/ompcc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omcpp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omc2x)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omcx2x)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omx2c)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omcnative)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omclinker)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/omc.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omf2f)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/ompf90)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omfpp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omf2x)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omfx2x)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omx2f)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omfnative)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omflinker)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/omf.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/xmpcc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/pp.conf.xmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/l2x.conf.xmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/x2x.conf.xmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/native.conf.xmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/linker.conf.xmp)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/xmpf90)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/pp.conf.xmpf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/l2x.conf.xmpf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/x2x.conf.xmpf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/native.conf.xmpf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/linker.conf.xmpf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/pp.conf.acc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/l2x.conf.acc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/x2x.conf.acc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/native.conf.acc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/linker.conf.acc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/setup.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omtranslate)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/omni_collect_init)
    AC_MSG_RESULT(yes)
else
    OMDRIVER_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(OMDRIVER_DIR)

#----
LIBXMP_HEADER_DIR=libxmp/include
LIBXMP_DIR=libxmp/src
LIBXMP_THREADS_DIR=libxmp/src_threads
LIBXMP_GPU_DIR=libxmp/src_gpu
AC_CONFIG_FILES($LIBXMP_HEADER_DIR/Makefile)
AC_CONFIG_FILES($LIBXMP_DIR/Makefile)
AC_CONFIG_FILES($LIBXMP_THREADS_DIR/Makefile)
AC_CONFIG_FILES($LIBXMP_GPU_DIR/Makefile)
AC_SUBST(LIBXMP_HEADER_DIR)
AC_SUBST(LIBXMP_DIR)
AC_SUBST(LIBXMP_THREADS_DIR)
AC_SUBST(LIBXMP_GPU_DIR)

#----
LIBXMPF_HEADER_DIR=libxmpf/include
LIBXMPF_DIR=libxmpf/src
#LIBXMPF_THREADS_DIR=libxmpf/src_threads
#LIBXMPF_GPU_DIR=libxmpf/src_gpu

AC_CONFIG_FILES($LIBXMPF_HEADER_DIR/Makefile)
AC_CONFIG_FILES($LIBXMPF_DIR/Makefile)
#AC_CONFIG_FILES($LIBXMPF_THREADS_DIR/Makefile)
#AC_CONFIG_FILES($LIBXMPF_GPU_DIR/Makefile)

AC_SUBST(LIBXMPF_HEADER_DIR)
AC_SUBST(LIBXMPF_DIR)
#AC_SUBST(LIBXMPF_THREADS_DIR)
#AC_SUBST(LIBXMPF_GPU_DIR)

# --------------------------------------------------------------------
# First of all, check on what system we are running.

#AC_CANONICAL_SYSTEM
AC_MSG_CHECKING(os)

HOSTARCH=`./buildutils/config.guess | grep -v Linking`
CPU=`echo $HOSTARCH | awk -F- '{ print $1 }' | sed 's/\./_/g'`
OS=`echo $HOSTARCH | awk -F- '{ print $3 }' | sed 's/\./_/g'`
PPTGT=""
MM5TGT=""

case $CPU in
    i*86)
        CPU=i386;;
    x86_64*)
        CPU=x86_64;;
    alpha*)
        CPU=alpha;;
esac

case $OS in
    sunos*)
        OS=sunos;;
    solaris*)
        OS=solaris;;
    freebsd*)
        OS=freebsd;;
    netbsd*)
        OS=netbsd;;
    irix*)
        OS=irix;;
    aix*)
        OS=aix;;
    darwin*)
        OS=darwin;;
    cygwin*)
        OS=cygwin32;;
esac

case $HOSTARCH in
    i*86-pc-linux*)
        PPTGT=linux;;
    alpha*-unknown-linux*)
        PPTGT=linux;;
    i386-pc-solaris*)
        PPTGT=sol2;;
    sparc-sun-solaris*)
        PPTGT=sol2;;
    mips-sgi-irix*)
        PPTGT=orig2k;;
esac

AC_MSG_RESULT($OS $CPU)
AC_SUBST(OS)
AC_SUBST(CPU)
AC_SUBST(MM5TGT)
AC_SUBST(PPTGT)

# --------------------------------------------------------------------
# OpenMP runtime checking

def_thread_type=""
def_lock_type=""
available_threads=""
available_locks=""
ignore_pair=""
ompcLibAsmSrcs=""

# Default thread type
# Check whether --with-thread or --without-thread was given.
AC_ARG_WITH(thread,
    [  --with-thread               type of openmp thread],
    [thread_type=$withval], [thread_type=""])

# Default lock type
# Check whether --with-lock or --without-lock was given.
AC_ARG_WITH(lock,
    [  --with-lock                 type of openmp lock],
    [lock_type=$withval], [lock_type=""])

case $OS in
    solaris*)
	def_thread_type=solaris
	def_lock_type=spin
	available_locks="mutex spin"
	available_threads="solaris pthread"
	case $CPU in
	    sparc*)
		ompcLibAsmSrcs="asm_sparc.s";;
	esac
	;;
    irix*)
	def_thread_type=sproc
	def_lock_type=spin
	available_locks="mutex spin"
	available_threads="pthread sproc"
	ignore_pair="sproc mutex"
	;;
    *)
	def_thread_type=pthread
	def_lock_type=mutex
	available_locks="mutex"
	available_threads="pthread"
	case $CPU in
	    i386*|mips*)
		def_lock_type=spin
		available_locks="mutex spin";;
	    sparc*)
		def_lock_type=spin
		available_locks="mutex spin"
		ompcLibAsmSrcs="asm_sparc.s";;
	    alpha*)
		def_lock_type=spin
		available_locks="mutex spin"
		ompcLibAsmSrcs="asm_alpha.s";;
	    rs6000*)
		available_locks="mutex spin";;
	    *)
		true;;
	esac
	;;
esac

if test "X${thread_type}" = "X"; then
    thread_type=${def_thread_type}
fi

if test "X${lock_type}" = "X"; then
    lock_type=${def_lock_type}
fi

OMP_THREAD_TYPE=${thread_type}
AC_SUBST(OMP_THREAD_TYPE)
OMP_LOCK_TYPE=${lock_type}
AC_SUBST(OMP_LOCK_TYPE)

case $lock_type in
    spin*)
	case $CPU in
	    i386*|x86_64*|mips*|sparc*|alpha*|rs6000*)
		true;;
	    *)
		echo "$ac_t""spin lock is only available on i386, x86_64, mips, alpha, sparc and rs6000 architectures." 1>&6
		exit 1;;
	esac;;
    *)
	true;;
esac

THREAD_DEF=""
case $thread_type in
    solaris*)
	THREAD_DEF="-DUSE_SOL_THREAD"
	;;
    sproc*)
	THREAD_DEF="-DUSE_SPROC"
	;;
    *)
	THREAD_DEF="-DUSE_PTHREAD"
	;;
esac

LOCK_DEF=""
case $lock_type in
    mutex*)
	LOCK_DEF=""
	;;
    spin*)
	LOCK_DEF="-DUSE_SPIN_LOCK"
	;;
esac

OMPC_LIB_ASM=${ompcLibAsmSrcs}

# --------------------------------------------------------------------
genLibTargetTmpl () {
    dir=$1
    tmpl=${dir}/target.tmpl
    echo "$ac_t""creating ${tmpl}" 1>&6
    > ${tmpl}
    ./buildutils/genLibTarget --locks="${available_locks}" --threads="${available_threads}" \
	--def_lock=${lock_type} --def_thread=${thread_type} --ignore_combo="${ignore_pair}" \
	--os=${OS} --cpu=${CPU} \
	`(cd ${dir};grep --regexp="^SRCS" Makefile.am | sed s/"SRCS		= "//g)` > ${tmpl}
}

#----
LIBOMPC_DIR="libompc/src"
AC_CONFIG_FILES($LIBOMPC_DIR/Makefile)
LIBOMPF_DIR="libompf/src"
AC_CONFIG_FILES($LIBOMPF_DIR/Makefile)
LIBTLOG_DIR="libtlog/src_threads"
AC_CONFIG_FILES($LIBTLOG_DIR/Makefile)
LIBTLOG_MPI_DIR="libtlog/src_mpi"
AC_CONFIG_FILES($LIBTLOG_MPI_DIR/Makefile)
AC_CONFIG_FILES($LIBTLOG_MPI_DIR/bin/tlogview)
genLibTargetTmpl $LIBOMPC_DIR

AC_SUBST(LIBOMPC_DIR)
AC_SUBST(LIBOMPF_DIR)
AC_SUBST(LIBTLOG_DIR)
AC_SUBST(LIBTLOG_MPI_DIR)

#----
LIBACC_DIR=libacc/src
LIBACC_HEADER_DIR=libacc/include
AC_CONFIG_FILES($LIBACC_DIR/Makefile)
AC_CONFIG_FILES($LIBACC_HEADER_DIR/Makefile)
AC_SUBST(LIBACC_DIR)
AC_SUBST(LIBACC_HEADER_DIR)

# --------------------------------------------------------------------
# check what CC will be used, and set compile flags

isGcc=yes
optCFlags=""
debCFlags=""
userBCFlags=""
mustCFlags=""
debCppFlags=""
userCppFlags=""
mustCppFlags=""
ccCom=""

AC_ARG_ENABLE(debug,
    [  --enable-debug                 enable generate executable with debug symbol (false default).],
    [doDebug=$enableval], [doDebug=no])

AC_ARG_ENABLE(xmpgpu,
    [  --enable-xmpgpu                enable XcalableMP-GPU (false default).],
    [doXmpGpu=$enableval], [doXmpGpu=no])

AC_ARG_ENABLE(openacc,
    [  --enable-openacc               enable OpenACC (false default).],
    [doOpenAcc=$enableval], [doOpenAcc=no])

AC_ARG_WITH(cppflag,
    [  --with-cppflag=OPT          C preprocessor options],
    [userCppFlags="${withval}"], [userCppFlags=""])

AC_ARG_WITH(backend-cflag,
    [  --with-backend-cflag=OPT    Backend C compiler options for optimization],
    [userBCFlags="${withval}"], [userBCFlags=""])

AC_ARG_ENABLE(atool,
    [  --enable-atool                 enable atool (false default).],
    [doatool=$enableval], [doatool=no])

AC_ARG_ENABLE(xacc,
    [  --enable-xacc                  enable XcalableACC (false default).],
    [xacc=$enableval], [xacc=no])

AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL

# --------------------------------------------------------------------
if test "x${doDebug}" = "xyes"; then
    debCFlags="-g"
    optCFlags="-O0"
    debCppFlags="-DDEBUG"
else
    debCFlags=""
    optCFlags="-O2"
    debCppFlags="-DNDEBUG"
fi

XMP_ENABLE_GPU=0
if test "x${doXmpGpu}" = "xyes"; then
    XMP_ENABLE_GPU=1
fi
AM_CONDITIONAL(ENABLE_XMP_GPU, test "x${doXmpGpu}" = "xyes")
AC_SUBST(XMP_ENABLE_GPU)

ENABLE_ACC=0
if test "x${doOpenAcc}" = "xyes"; then
    ENABLE_ACC=1
fi
AM_CONDITIONAL(ENABLE_ACC, test "x${doOpenAcc}" = "xyes")
AC_SUBST(ENABLE_ACC)
 
ENABLE_ATOOL=0
if test "x${doatool}" = "xyes"; then
   ENABLE_ATOOL=1
fi
AM_CONDITIONAL(ENABLE_ATOOL, test "x${doatool}" = "xyes")
AC_SUBST(ENABLE_ATOOL)

#ENABLE_XACC=0
#if test "x${xacc}" = "xyes"; then
#   ENABLE_XACC=1
#fi
AM_CONDITIONAL(ENABLE_XACC, test "x${xacc}" = "xyes")
AC_SUBST(ENABLE_XACC)

# check gcc version
AC_MSG_CHECKING(gcc version)
changequote(<<, >>)
gccVer=`${CC} --version 2>&1 | head -n1 | sed 's:[^ ]* [^ ]* ::' | sed 's: .*::'`
changequote([, ])
AC_MSG_RESULT($gccVer)

case $gccVer in
2*|3*|4.0*|4.1.0|4.1.1|4.1.2)
	;;
4*)
    mustCFlags="-fgnu89-inline"
	;;
esac

mustCFlags="${mustCppFlags}"

# Generate cpu/os specifier
CPUDEF=-DOMNI_CPU_`echo $CPU | tr '[a-z]' '[A-Z]'`
OSDEF=-DOMNI_OS_`echo $OS | tr '[a-z]' '[A-Z]'`

if test "$target" != "$host"; then
   TARGET_CPU=`echo $target | awk -F- '{ print $1 }' | sed 's/\./_/g'`
   TARGET_CPUDEF=-DOMNI_TARGET_CPU_`echo ${TARGET_CPU} | tr '[a-z]' '[A-Z]'`
fi

CPPFLAGS="${CPUDEF} ${OSDEF} ${debCppFlags} ${userCppFlags} ${mustCppFlags} ${TARGET_CPUDEF}"
AC_SUBST(CPPFLAGS)

CFLAGS="${debCFlags} ${optCFlags} ${mustCFlags} ${CFLAGS}"
CXXFLAGS=$CFLAGS
AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)

AH_TEMPLATE(OBJEXT, [object file extension])
AC_DEFINE_UNQUOTED(OBJEXT, "${OBJEXT}")

# --------------------------------------------------------------------
# Specify Backend Compiler and Options
IS_CRAY="No"
IS_AIX="No"
AIX_OPT=""
if test "$target" != "$host"; then
  if test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu"; then
    BECC="mpifccpx"
    BEFC="mpifrtpx"
    BECC_OPTION="-Xg -noansi -Kfast"
    BEFC_MODINC="-M"
  elif test "$target" = "MIC-linux-gnu"; then
    BECC="mpiicc"
    BEFC="mpiifort"
    BEFC_MODINC="-module"
  elif test "$target" = "Cray-linux-gnu"; then
    BECC="cc"
    BEFC="ftn"
    BECC_OPTION="-hgnu"
    IS_CRAY="Yes"
    BEFC_MODINC="-J"
  elif test "$target" = "sx-nec-superux"; then
    BECC="sxmpic++"
    BEFC="sxmpif90"
    BECC_OPTION=""
    BEFC_MODINC="-to"
  elif test "$target" = "powerpc-ibm-none"; then
    BECC="mpixlcxx_r"
    BEFC="mpixlf90_r"
    BEFC_OPTION="-qextname"
    BECC_OPTION=""
    BEFC_MODINC="-I"
  elif test "$target" = "powerpc64-hitachi-aix"; then
    BECC="mpCC_r"
    BEFC="mpxlf90_r"
    BEFC_OPTION="-qextname -q64"
    BECC_OPTION="-qcpluscmt -q64"
    AIX_OPT="-maix64"
    IS_AIX="Yes"
    BEFC_MODINC="-I"
  fi
else  # if "--target" option is not used in ./configure
  BECC="mpicc"
  BEFC="mpif90"
  BEFC_MODINC="-M"
fi

PPCC="${BECC}"
PPCC_OPT="-E"
NATIVE_DOPT="-D_XCALABLEMP"

if test "$target" = "powerpc-ibm-none"; then
  PPCC="gcc"
  PPCC_OPT="-E --std=c99"
  NATIVE_DOPT="-WF,-D_XCALABLEMP"
fi

if test "$target" = "powerpc64-hitachi-aix"; then
  NATIVE_DOPT="-WF,-D_XCALABLEMP"
fi

AC_SUBST(PPCC)
AC_SUBST(PPCC_OPT)
AC_SUBST(NATIVE_DOPT)

AC_SUBST(AIX_OPT)
if test "$IS_AIX" = "Yes"; then
  AC_DEFINE(AIX_64BIT)
  AH_TEMPLATE(AIX_64BIT, [aix int64_t])
fi

# Add temporarily by mnakao
enableKRDMA=""
AC_ARG_ENABLE(KRDMA,
   [  --enable-KRDMA                 enable RDMA functions on the K compiter.],
   [enableKRDMA=$enableval], [enableKRDMA=no])

AM_CONDITIONAL(FJRDMA, test x${enableKRDMA} = xyes)
AC_SUBST(FJRDMA)
# End Add temporarily

#AM_CONDITIONAL(FJRDMA, test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu")
#AC_SUBST(FJRDMA)

# --------------------------------------------------------------------
# specify Backend C compiler
AC_ARG_WITH(backend-cc,
    [  --with-backend-cc=CMD       Backend C compiler (default: mpicc)],
    [BECC="${withval}"], [BECC="${BECC}"])

# --------------------------------------------------------------------
# specify Backend F90 compiler
AC_ARG_WITH(backend-fc,
    [  --with-backend-fc=CMD       Backend F90 compiler (default: mpif90)],
    [BEFC="${withval}"], [BEFC="${BEFC}"])

# --------------------------------------------------------------------
AC_SUBST(BECC)
AC_SUBST(BEFC)

AC_SUBST(BEFC_MODINC)

AC_SUBST(IS_CRAY)
AC_SUBST(IS_AIX)

AM_CONDITIONAL(SR16000, test "$target" = "powerpc64-hitachi-aix") 
AC_SUBST(SR16000)

AM_CONDITIONAL(SUPERUX, test "$target" = "sx-nec-superux")
AC_SUBST(SUPERUX)

#AM_CONDITIONAL(CRAY, test "$target" = "Cray-linux-gnu")
#AC_SUBST(CRAY)

# --------------------------------------------------------------------
# Specify Options of Backend C Compiler when compiling XMP code
IS_PGCC="No"
AC_MSG_CHECKING([backend C compiler])
if test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu"; then
  OMC_CONF_LX2X_OPT=""
  OMC_CONF_NTV_OPT="-Xg -noansi"
  OMC_CONF_LNK_OPT="-Kfast,openmp"
  BECC_OMP_OPTION="-Kfast,openmp"
elif test "$target" = "Cray-linux-gnu"; then
  OMC_CONF_LX2X_OPT=""
  OMC_CONF_NTV_OPT="-hgnu -D_XMP_CRAY"
  OMC_CONF_LNK_OPT=""
  BECC_OMP_OPTION=""
elif test "$target" = "sx-nec-superux"; then
  OMC_CONF_LX2X_OPT=""
  OMC_CONF_NTV_OPT="-Kgcc -w"
  OMC_CONF_LNK_OPT="-Popenmp"
  BECC_OMP_OPTION="-Popenmp"
elif test "$target" = "powerpc-ibm-none"; then
  OMC_CONF_LX2X_OPT=""
  OMC_CONF_NTV_OPT="-q64"
  DRIVER="/bgsys/drivers/ppcfloor"
  OMC_CONF_LNK_OPT="-I/usr/local/bg/include \
                -L/usr/local/bg/lib -lkek -L${DRIVER}/bgpm/lib/ -lbgpm -g -O3 -qhot -qsmp=omp -qsimd=auto -q64"
  BECC_OPTION="-O5"
  BECC_OMP_OPTION="-O5  -qhot -qsmp=omp -qsimd=auto"
elif test "$target" = "powerpc64-hitachi-aix"; then
  OMC_CONF_LX2X_OPT=""
  OMC_CONF_NTV_OPT="-q64"
  OMC_CONF_LNK_OPT="-q64 -qhot -qsmp=omp -qsimd=auto"
  BECC_OPTION="-O5"
  BECC_OMP_OPTION="-O5 -qhot -qsmp=omp -qsimd=auto"
else
  chk=`$BECC --version | head -n1`
  case $chk in
  cc* | gcc* | Apple*)
    OMC_CONF_LX2X_OPT=""
    OMC_CONF_NTV_OPT="-O2 -std=gnu99 -Wno-implicit -Wno-attributes"
    OMC_CONF_LNK_OPT="-O2 -fopenmp"
    BECC_OPTION="-O2 -std=gnu99 -Wall -Wpointer-arith"
    BECC_OMP_OPTION="-O2 -fopenmp"
    ;;
  *ICC*)
    OMC_CONF_LX2X_OPT=""
    OMC_CONF_NTV_OPT="-std=gnu99 -w"
    OMC_CONF_LNK_OPT="-openmp"
    BECC_OPTION="-std=gnu99 -Wall -Wpointer-arith"
    BECC_OMP_OPTION="-openmp"
    ;;
  *)
    chk2=`$BECC --version 2> /dev/null | head -n2 | tail -n1`
    case $chk2 in
    pgcc*)
      OMC_CONF_LX2X_OPT=""
      OMC_CONF_NTV_OPT="-O2 -D_XMP_PGI"
      OMC_CONF_LNK_OPT="-O2 -mp"
      BECC_OPTION="-O2 -c99 -Minform=warn"
      BECC_OMP_OPTION="-O2 -mp"
      IS_PGCC="Yes"
      ;;
    *)
    echo "No supported C compiler was found."
    echo "This program supports following C compilers."
    echo " - GNU C Compiler (mpicc)"
    echo " - Intel C Compiler (mpiicc)"
    echo " - Fujitsu C Compiler (mpifccpx)"
    echo " - Cray C Compiler (cc)"
    echo " - NEC C/C++ Compiler (sxc++)"
    echo " - IBM C Compiler (mpixlcxx_r)"
    echo " - HITACHI C Compiler (mpCC_r)"
    echo " - PGI C Compiler (pgcc)"
exit 1
    ;;
    esac 
    ;;
  esac
fi

BECC_OPTION="${BECC_OPTION} ${userBCFlags}"

AC_SUBST(OMC_CONF_LX2X_OPT)
AC_SUBST(OMC_CONF_NTV_OPT)
AC_SUBST(OMC_CONF_LNK_OPT)
AC_SUBST(BECC_OMP_OPTION)
AC_SUBST(BECC_OPTION)
AC_SUBST(IS_PGCC)

# --------------------------------------------------------------------
# Specify Options of Backend Fortran Compiler
if test $enableF2F = yes; then
  AC_PROG_FC([${FC},gfortran,ifort])
  AC_MSG_CHECKING([backend Fortran compiler])

  if test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu"; then
    OMF_CONF_LX2X_OPT=""
    OMF_CONF_NTV_OPT="-Free -Kfast"
    OMF_CONF_LNK_OPT="-Kfast,openmp"
  elif test "$target" = "Cray-linux-gnu"; then
    OMF_CONF_LX2X_OPT=""
    OMF_CONF_NTV_OPT="-f free"
    OMF_CONF_LNK_OPT=""
  elif test "$target" = "sx-nec-superux"; then
    OMF_CONF_LX2X_OPT=""
    OMF_CONF_NTV_OPT="-f4 -w"
    OMF_CONF_LNK_OPT="-Popenmp"
  elif test "$target" = "powerpc-ibm-none"; then
    OMF_CONF_LX2X_OPT=""
    OMF_CONF_NTV_OPT="-qextname -O5 -q64"
    OMF_CONF_LNK_OPT="-I/usr/local/bg/include -g -O3 -qhot -qsmp=omp -qsimd=auto -q64 -qsource -qlist"
  elif test "$target" = "powerpc64-hitachi-aix"; then
    OMF_CONF_LX2X_OPT=""
    OMF_CONF_NTV_OPT="-qextname -qnomixed -q64  -O5"
    OMF_CONF_LNK_OPT="-q64  -O5 -qhot -qsmp=omp -qsimd=auto"
  else
    chk=`$BEFC --version 2> /dev/null | head -n1`
    case $chk in
    GNU*)
      OMF_CONF_LX2X_OPT="-gnu"
      OMF_CONF_NTV_OPT="-ffree-form"
      OMF_CONF_LNK_OPT="-fopenmp -lgomp"
      BEFC_MODINC="-J"  
      ;;
    *IFORT*)
      OMF_CONF_LX2X_OPT="-intel"
      OMF_CONF_NTV_OPT="-auto"
      OMF_CONF_LNK_OPT="-threads -openmp"
      BEFC_MODINC="-module"
      ;;
    *)
      chk2=`$BEFC --version 2>/dev/null | head -n2 | tail -n1`
      case $chk2 in
      pgfortran*|pgf95*|pgf90*)
        OMF_CONF_LX2X_OPT=""
        OMF_CONF_NTV_OPT="-O2"
        OMF_CONF_LNK_OPT="-O2 -mp"
	;;
      *)
      echo "No supported Fortran compiler was found."
      echo "This program supports following Fortran compilers."
      echo " - GNU Fortran (mpif77 or mpif90)"
      echo " - Intel Fortran (mpiifort)"
      echo " - Fujitsu Fortran (mpifrtpx)"
      echo " - Cray Fortran (ftn)"
      echo " - IBM Fortran (mpixlf90_r)"
      echo " - Hitachi Fortran (mpif90)"
      echo " - PGI Fortran (mpif90)"
      exit 1
      ;;
    esac
      ;;
    esac
  fi
  AC_SUBST(OMF_CONF_LX2X_OPT)
  AC_SUBST(OMF_CONF_NTV_OPT)
  AC_SUBST(OMF_CONF_LNK_OPT)
  AC_SUBST(BEFC_OPTION)
  AC_SUBST(BEFC_MODINC)
  AC_PROG_FC_C_O
fi

# --------------------------------------------------------------------
# Various command check

# check make programs
AC_PROG_MAKE_SET

# check awk
AC_PROG_AWK

# check bash
BASH=""
AC_PATH_PROGS(BASH, bash, ${PATH}:/bin:/usr/bin:/usr/local/bin:/opt/bin:/opt/local/bin)
AC_SUBST(BASH)

RANLIB=""
AR=""
NM=""

# check ranlib & ar & nm
if test "$target" != "$host"; then
   case "$target" in
	Kcomputer-linux-gnu|FX10-linux-gnu)
#		AC_CHECK_TOOL([RANLIB], [ranlibpx], [:])
		AC_CHECK_PROG([RANLIB], [ranlibpx], [ranlibpx], [:])
		if test "${RANLIB}" = ":"; then
		    AC_MSG_ERROR([ranlibpx not found]);
		fi		    

#		AC_CHECK_TOOL([AR], [arpx], [:])
		AC_CHECK_PROG([AR], [arpx], [arpx], [:])
		if test "${AR}" = ":"; then
		    AC_MSG_ERROR([arpx not found]);
		fi		    

		ARFLAGS="cru"
		AC_SUBST(ARFLAGS)

#		AC_CHECK_TOOL([NM], [nmpx], [:])
		AC_CHECK_PROG([NM], [nmpx], [nmpx], [:])
		if test "${AR}" = ":"; then
		    AC_MSG_ERROR([nmpx not found]);
		fi ;;
	sx-nec-superux)
#		AC_CHECK_TOOL([RANLIB], [echo], [echo], [:])
		AC_CHECK_PROG([RANLIB], [echo], [echo], [:])
#		if test "${RANLIB}" = ":"; then
#		    AC_MSG_ERROR([sxranlib not found]);
#		fi		    

#		AC_CHECK_TOOL([AR], [sxar], [:])
		AC_CHECK_PROG([AR], [sxar], [sxar], [:])
		if test "${AR}" = ":"; then
		    AC_MSG_ERROR([sxar not found]);
		fi		    

		ARFLAGS="cru"
		AC_SUBST(ARFLAGS)

#		AC_CHECK_TOOL([NM], [sxnm], [:])
		AC_CHECK_PROG([NM], [sxnm], [sxnm], [:])
		if test "${NM}" = ":"; then
		    AC_MSG_ERROR([sxnm not found]);
		fi ;;
	powerpc64-hitachi-aix)
#                AC_CHECK_TOOL([RANLIB], [echo], [:])
                AC_CHECK_PROG([RANLIB], [echo], [echo], [:])
#                if test "${RANLIB}" = ":"; then
#                    AC_MSG_ERROR([ranlib not found]);
#                fi

#                AC_CHECK_TOOL([AR], [ar], [:])
                AC_CHECK_PROG([AR], [ar], [ar], [:])
                if test "${AR}" = ":"; then
                    AC_MSG_ERROR([ar not found]);
                fi

                ARFLAGS="-X32_64 cru"
                AC_SUBST(ARFLAGS)

#                AC_CHECK_TOOL([NM], [nm], [:])
                AC_CHECK_PROG([NM], [nm], [nm], [:])
                if test "${NM}" = ":"; then
                    AC_MSG_ERROR([nm not found]);
                fi ;;         
	powerpc-ibm-none)
#                AC_CHECK_TOOL([RANLIB], [echo], [:])
                AC_CHECK_PROG([RANLIB], [echo], [echo], [:])
#                if test "${RANLIB}" = ":"; then
#                    AC_MSG_ERROR([bgranlib not found]);
#                fi

#                AC_CHECK_TOOL([AR], [ar], [:])
                AC_CHECK_PROG([AR], [ar], [ar], [:])
                if test "${AR}" = ":"; then
                    AC_MSG_ERROR([bgar not found]);
                fi

                ARFLAGS="cru"
                AC_SUBST(ARFLAGS)

#                AC_CHECK_TOOL([NM], [nm], [:])
                AC_CHECK_PROG([NM], [nm], [nm], [:])
                if test "${NM}" = ":"; then
                    AC_MSG_ERROR([nm not found]);
                fi ;;         
	Cray-linux-gnu)
		AC_PROG_RANLIB
#		AC_CHECK_TOOL([AR], [ar], [:])
		AC_CHECK_PROG([AR], [ar], [ar], [:])
		ARFLAGS="cru"
		AC_SUBST(ARFLAGS)
#		AC_CHECK_TOOL([NM], [nm], [:])
		AC_CHECK_PROG([NM], [nm], [nm], [:])
                ;;
	*)
		AC_CHECK_TARGET_TOOL([RANLIB], [ranlib], [:])
		AC_CHECK_TARGET_TOOL([AR], [ar], [:])
		AC_CHECK_TARGET_TOOL([NM], [nm], [:]) ;;
   esac
else
   AC_PROG_RANLIB
   if test "x${isGcc}" = "xno"; then
      case $OS in
          solaris*|irix*)
	  RANLIB=':';;
      esac
   fi

   AC_CHECK_TOOL([AR], [ar], [:])
   ARFLAGS="cru"
   AC_SUBST(ARFLAGS)

   AC_CHECK_TOOL([NM], [nm], [:])

fi

# check CC for libraries
if test "$target" != "$host"; then
   case "$target" in
	Kcomputer-linux-gnu|FX10-linux-gnu)
		CC_LIB="fccpx -Xg -std=gnu99"
		AC_SUBST(CC_LIB) ;;
        MIC-linux-gnu)
		CC_LIB="icc -mmic"
		AC_SUBST(CC_LIB) ;;
	sx-nec-superux)
		CC_LIB="sxc++ -Kgcc"
		AC_SUBST(CC_LIB) ;;
	powerpc64-hitachi-aix)
		CC_LIB="lxc -qgcc_c_stdinc"
		AC_SUBST(CC_LIB);;
	powerpc-ibm-none)
		CC_LIB="bglxc -qgcc_c_stdinc"
		AC_SUBST(CC_LIB);;
	*)
		AC_CHECK_TARGET_TOOL([CC_LIB], [cc], [:]) ;;
   esac

else

   CC_LIB=${CC}
   AC_SUBST(CC_LIB)

fi

# check zip
ZIPCOM=""
AC_PATH_PROG(ZIPCOM, zip, zip, ${PATH}:/bin:/usr/bin:/usr/local/bin:/opt/bin:/opt/local/bin)
AC_SUBST(ZIPCOM)

# check Perl versoin is 5
PERL5=""
AC_PATH_PROGS(PERL5, perl)

inDev=0
if test "$PERL5" = ""; then
    inDev=0
else
    inDev=1
    isPerlVersion5=no
    AC_MSG_CHECKING(Perl version is 5)
    pTmp=/tmp/.btoolPerlChk.$$
    ${PERL5} -e 'require 5;' > ${pTmp} 2>&1
    if test "$?" = "0"; then
        AC_MSG_RESULT([yes])
        isPerlVersion5=yes
    else
        AC_MSG_RESULT([no])
        isPerlVersion5=no
    fi
    rm -f ${pTmp}
    unset chk pTmp

fi
AC_SUBST(PERL5)

# check lex and yacc

AC_PROG_LEX
#AC_CHECK_PROGS([LEX], [flex]) 
if test $LEX = ":"; then
exit 1
fi
AC_PROG_YACC

# --------------------------------------------------------------------
# check java

useJava=yes

JAVACOM=""
AC_PATH_PROG(JAVACOM, java, java, ${JAVA_HOME}/bin)
JAVA_HOME=`$JAVACOM -classpath buildutils javahome`
java_home_base=`basename ${JAVA_HOME}`

if test "${java_home_base}" = "jre"; then
    JAVA_HOME=`dirname ${JAVA_HOME}`
fi

AC_SUBST(JAVA_HOME)

JAVACCOM=""
AC_PATH_PROG(JAVACCOM, javac, javac, ${JAVA_HOME}/bin:${PATH}:/usr/java/default/bin:/bin:/usr/bin:/usr/local/bin:/opt/bin:/opt/local/bin)

JARCOM=""
AC_PATH_PROG(JARCOM, jar, jar, ${JAVA_HOME}/bin)

AC_SUBST(JAVACOM)
AC_SUBST(JAVACCOM)
AC_SUBST(JARCOM)

# check Java VM is IBM Java6

isIBMJava6=no
if test "x${useJava}" = "xyes"; then
    AC_MSG_CHECKING([IBM Java6 or later])
    jTmp=/tmp/.omniJavaChk.$$
    ${JAVACOM} -fullversion > ${jTmp} 2>&1
    chk=`grep 'JRE 1.6.0 IBM' ${jTmp}` || `grep 'JRE 1.7.0 IBM' ${jTmp}` || `grep 'JRE 1.8.0 IBM' ${jTmp}`
    if test "x${chk}" = "x"; then
        AC_MSG_RESULT([no])
    else
        AC_MSG_RESULT([yes])
        isIBMJava6=yes
    fi
    rm -f ${jTmp}
    unset chk jTmp
fi
AC_SUBST(isIBMJava6)

# check Java VM version is Sun Java6
isSunJava6=no
if test "x${useJava}" = "xyes"; then
    AC_MSG_CHECKING([Sun Java6 or later])

    changequote(<<, >>)
    case $target in
      powerpc-ibm-none|powerpc64-hitachi-aix)
        chk=`${JAVACOM} -fullversion 2>&1 | sed -e 's:openjdk[\t ]*full[\t ]*version[\t ]*::g' -e 's:\"::g'`
      ;;
	*)
        chk=`${JAVACOM} -fullversion 2>&1 | sed -e 's:java[\t ]*full[\t ]*version[\t ]*::g' -e 's:\"::g'`
	;;
    esac 
    changequote([, ])

    if test "x${chk}" = "x"; then
        AC_MSG_RESULT([no])
    fi
    case $chk in
    1.6*|1.7*|1.8*)
        isSunJava6=yes
        AC_MSG_RESULT([yes]);;
    *)
        isSunJava6=no
        AC_MSG_RESULT([no]);;
    esac
    unset chk
fi

if test "${isSunJava6}" = "no" -a "${isIBMJava6}" = "no"; then
    AC_MSG_ERROR([Java6 or later not found]);
fi

AC_SUBST(isSunJava6)

# check java type
javaType=unknown
if test "x${useJava}" = "xyes"; then
    if test "x${isIBMJava6}" = "xyes"; then
        javaType=ibm
    elif test "x${isSunJava6}" = "xyes"; then
        javaType=sun
    fi
fi
AC_SUBST(javaType)

# check ant
ANTCOM=""
AC_PATH_PROG(ANTCOM, ant, , ${PATH}:/bin:/usr/bin:/usr/local/bin:/opt/bin:/opt/local/bin)

if test "$ANTCOM" = ""; then
    AC_MSG_ERROR([requires Apache Ant])
fi

ANTCOM="JAVA_HOME=$JAVA_HOME ${ANTCOM}"
AC_SUBST(ANTCOM)

# check relaxer
#relaxerDir=/usr/local/relaxer
#AC_MSG_CHECKING([for relaxer])
#AC_ARG_WITH(relaxerDir,
#    [  --with-relaxerDir=DIR  specify where the Relaxer is installed.],
#    [relaxerDir=$withval], [relaxerDir=/usr/local/relaxer])
#
#if test -e "${relaxerHome}/Relaxer.jar" ; then
#    AC_MSG_RESULT($relaxerDir)
#    RELAXER_DIR="${relaxerDir}"
#else
#    AC_MSG_RESULT(no)
#    RELAXER_DIR=""
#fi
#
#AC_SUBST(RELAXER_DIR)

# set java library install dir to install target of java_DATA
javadir='${prefix}/share/xcalablemp'
AC_SUBST(javadir)

# java option
JAVA_OPT="-Xmx1024m"
AC_SUBST(JAVA_OPT)

#--------------------------------------------------------------------
# endian check
AC_C_BIGENDIAN

# --------------------------------------------------------------------
# Checks for types
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_MODE_T
AC_TYPE_PID_T
AC_TYPE_UID_T
AC_STRUCT_TM

# --------------------------------------------------------------------
# Checks for header files

# check stdlib.h, stdarg.h, string.h, float.h
AC_HEADER_STDC
# check if needed sys/time.h with time.h
AC_HEADER_TIME

AC_CHECK_HEADERS([fcntl.h libintl.h malloc.h stddef.h stdint.h limits.h strings.h unistd.h errno.h locale.h sys/time.h sys/resource.h sys/wait.h sys/stat.h assert.h])

# --------------------------------------------------------------------
# Checks for library functions.

if test "$target" != "$host"; then
   case "$target" in
        powerpc64-hitachi-aix);;
        *)
	AC_FUNC_MALLOC;;
   esac
else
	AC_FUNC_MALLOC
fi

#AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_FORK
AC_FUNC_STAT
AC_FUNC_VPRINTF
AC_FUNC_STRFTIME
AC_FUNC_ERROR_AT_LINE
AC_FUNC_STRTOD

AC_CHECK_FUNCS([bzero dup2 memset setlocale system setenv])
AC_CHECK_FUNCS([strcasecmp strchr strdup strncasecmp strrchr strstr strtol])

AC_CHECK_LIB(m, pow)
AC_CHECK_LIB(m, sqrt)

# C99 functions
AC_CHECK_FUNC(strtoul, , AC_MSG_ERROR([strtoul not found]))
AC_CHECK_FUNC(strtoll, , AC_MSG_ERROR([strtoll not found]))
AC_CHECK_FUNC(strtoull, , AC_MSG_ERROR([strtoull not found]))
AC_CHECK_FUNC(strtold, , AC_MSG_ERROR([strtold not found]))

#--------------------------------------------------------------------
#    The code below deals with several issues related to gettimeofday:
#    1. Some systems don't provide a gettimeofday function at all
#       (set NO_GETTOD if this is the case).
#    2. SGI systems don't use the BSD form of the gettimeofday function,
#       but they have a BSDgettimeofday function that can be used instead.
#    3. See if gettimeofday is declared in the <sys/time.h> header file.
#       if not, set the GETTOD_NOT_DECLARED flag so that tclPort.h can
#       declare it.
#--------------------------------------------------------------------
AC_CHECK_FUNCS(gettimeofday, , [AC_DEFINE(NO_GETTOD)])
AH_TEMPLATE(NO_GETTOD, [no gettimeofday])
AC_EGREP_HEADER(gettimeofday, sys/time.h, , AC_DEFINE(GETTOD_NOT_DECLARED))
AH_TEMPLATE(GETTOD_NOT_DECLARED, [not declared gettimeofday in sys/time.h])

# --------------------------------------------------------------------
# check type size/align

AC_CHECK_SIZEOF(unsigned char)
AC_CHECK_SIZEOF(unsigned short)
AC_CHECK_SIZEOF(unsigned int)
AC_CHECK_SIZEOF(unsigned long)
AC_CHECK_SIZEOF(unsigned long long)
AC_CHECK_SIZEOF(float)
AC_CHECK_SIZEOF(double)
AC_CHECK_SIZEOF(long double)
AC_CHECK_SIZEOF(void *)
AC_CHECK_SIZEOF(_Bool)

SIZEOF_DOUBLE=$ac_cv_sizeof_double
AC_SUBST(SIZEOF_DOUBLE)
SIZEOF_FLOAT=$ac_cv_sizeof_float
AC_SUBST(SIZEOF_FLOAT)
SIZEOF_LONG_DOUBLE=$ac_cv_sizeof_long_double
AC_SUBST(SIZEOF_LONG_DOUBLE)
SIZEOF_UNSIGNED_CHAR=$ac_cv_sizeof_unsigned_char
AC_SUBST(SIZEOF_UNSIGNED_CHAR)
SIZEOF_UNSIGNED_INT=$ac_cv_sizeof_unsigned_int
AC_SUBST(SIZEOF_UNSIGNED_INT)
SIZEOF_UNSIGNED_LONG=$ac_cv_sizeof_unsigned_long
AC_SUBST(SIZEOF_UNSIGNED_LONG)
SIZEOF_UNSIGNED_LONG_LONG=$ac_cv_sizeof_unsigned_long_long
AC_SUBST(SIZEOF_UNSIGNED_LONG_LONG)
SIZEOF_UNSIGNED_SHORT=$ac_cv_sizeof_unsigned_short
AC_SUBST(SIZEOF_UNSIGNED_SHORT)
SIZEOF_VOID_P=$ac_cv_sizeof_void_p
AC_SUBST(SIZEOF_VOID_P)
SIZEOF__BOOL=$ac_cv_sizeof__Bool
AC_SUBST(SIZEOF__BOOL)

AC_CHECK_ALIGNOF(unsigned char)
AC_CHECK_ALIGNOF(unsigned short)
AC_CHECK_ALIGNOF(unsigned int)
AC_CHECK_ALIGNOF(unsigned long)
AC_CHECK_ALIGNOF(unsigned long long)
AC_CHECK_ALIGNOF(float)
AC_CHECK_ALIGNOF(double)
AC_CHECK_ALIGNOF(long double)
AC_CHECK_ALIGNOF(void *)
AC_CHECK_ALIGNOF(_Bool)

# --------------------------------------------------------------------
# 32/64 bit int type

rm -rf ./szchk ./szchk.*
AC_MSG_RESULT([Creating size check program])
${CC} ${CPPFLAGS} ${CFLAGS} ${hasLLFlags} ./buildutils/szchk.c -o ./szchk
if test ! -x ./szchk; then
    AC_MSG_RESULT([can't create size check program. exit.])
    exit 1
fi

F2C_INT64_FLAGS=""
for i in 16 32 64
do
    AC_MSG_CHECKING([$i bit integer])
    typName=`./szchk $i`
    if test "x${typName}" = "xunknown"; then
    case $i in
    16)
        TYPE_INT16="short"
        AC_SUBST(TYPE_INT16)
        asump=${TYPE_INT16};;
    32)
        TYPE_INT32="int"
        AC_SUBST(TYPE_INT32)
        asump=${TYPE_INT32};;
    64)
        TYPE_INT64="long long int"
        AC_SUBST(TYPE_INT64)
        asump=${TYPE_INT64};;
    esac
    AC_MSG_RESULT([not supported. use $asump])
    else
    AC_MSG_RESULT([$typName])
    case $i in
    16)
        AC_DEFINE(HAS_INT16)
        AH_TEMPLATE(HAS_INT16, [has int16])
        TYPE_INT16=${typName}
        AC_SUBST(TYPE_INT16);;
    32)
        AC_DEFINE(HAS_INT32)
        AH_TEMPLATE(HAS_INT32, [has int32])
        TYPE_INT32=${typName}
        AC_SUBST(TYPE_INT32);;
    64)
        AC_DEFINE(HAS_INT64)
        AH_TEMPLATE(HAS_INT64, [has int64])
        F2C_INT64_FLAGS="-DAllow_TYQUAD"
        TYPE_INT64=${typName}
        AC_SUBST(TYPE_INT64);;
    esac
    fi
done
AC_SUBST(F2C_INT64_FLAGS)
rm -rf ./szchk ./szchk.*

# --------------------------------------------------------------------
# check size of integer enough to hold size of void *

AC_MSG_CHECKING([integer type enough to hold void pointer])
${CC} -I. ${CPPFLAGS} ${CFLAGS} ${hasLLFlags} ./buildutils/chkvoidp.c -o chkvoidp
voidPint="unknown"
if test -x ./chkvoidp; then
    voidPint=`./chkvoidp`
else
    AC_MSG_RESULT([can't create check program. exit.])
    exit 1
fi
if test "x${voidPint}" = "xunknown"; then
    AC_MSG_RESULT([can't determine????])
    exit 1
else
    AC_MSG_RESULT([$voidPint])
fi
INT_ENUFF_FOR_VOIDP=$voidPint
AC_SUBST(INT_ENUFF_FOR_VOIDP)
rm -f ./chkvoidp.* ./chkvoidp

# --------------------------------------------------------------------
# check 64 bit virtual address

AC_MSG_CHECKING([64 bit virtual address])
${CC} -I. ${CPPFLAGS} ${CFLAGS} ./buildutils/addr64.c -o ./addr64
addrIs64="unknown"
if test -x ./addr64; then
    addrIs64=`./addr64`
else
    AC_MSG_RESULT([can't create check program. exit.])
    exit 1
fi
if test "x$addrIs64" = "xunknown"; then
    AC_MSG_RESULT([can't determine????])
    exit 1
else
    if test "x$addrIs64" = "x8"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(ADDR_IS_64)
        AH_TEMPLATE(ADDR_IS_64, [])
    else
        AC_MSG_RESULT([no])
    fi
fi
rm -f ./addr64.* ./addr64

# --------------------------------------------------------------------
# has printf "%qd"

rm -f ./hasqd.* ./hasqd
AC_MSG_CHECKING([has quad-precision print format])
${CC} -I. ${CPPFLAGS} ${CFLAGS} ./buildutils/hasqd.c -o ./hasqd > /dev/null 2>&1
hasQD=no
if test -x ./hasqd; then
    ret=`./hasqd`
    if test "x$ret" = "xqd" -o "x$ret" = "x%qd"; then
        hasQD=no
    else
        hasQD=yes
    fi
fi
AC_MSG_RESULT([${hasQD}])
if test "x${hasQD}" = "xyes"; then
    AC_DEFINE(HAVE_QUAD_PRINT)
    AH_TEMPLATE(HAVE_QUAD_PRINT, [has printf "%qd"])
fi
rm -f ./hasqd.* ./hasqd

# --------------------------------------------------------------------
# quad real support

gmpInclude=/usr/include
gmpLib=/usr/lib
gmpIncOpt=""
gmpDirOpt=""
qRealPrecBits=128

AC_ARG_ENABLE(mreal,
    [  --enable-mreal                 enable multiple precision real (and real*16) support of Fortran 77 (false default).],
    [doQReal=$enableval], [doQReal=no])

AC_ARG_WITH(gmp-lib,
    [  --with-gmp-lib=DIR          where the GNU-MP library is for quad-real support.],
    [gmpLib=$withval], [true])

AC_ARG_WITH(gmp-include,
    [  --with-gmp-include=DIR      where the GNU-MP include headers are for quad-real support.],
    [gmpInclude=$withval], [true])

if test "x${doQReal}" = "xyes"; then
    if test -f ${gmpInclude}/gmp.h; then
        true
    else
        AC_MSG_RESULT([can't find ${gmpInclude}/gmp.h. exit.])
        exit 1
    fi
    gmpLibs=`ls ${gmpInclude}/libgmp.* 2> /dev/null`
    if test "x${gmpLibs}" = "x"; then
        AC_MSG_RESULT([can't find ${gmpLib}/libgmp.*. exit.])
        exit 1
    fi
    gmpIncOpt="-I${gmpInclude}"
    gmpLibOpt="-L${gmpLib} -lgmp"
fi

AC_ARG_WITH(precision,
    [  --with-precision=NUM        the default precision of real*16 in bits (128).],
    [qRealPrecBits=$withval], [qRealPrecBits=128])


# libgmp check
omniLimbT="int"
omniExpT="int"
omniQRealPrecInLimb="1"
omniQRealFillGap=""
if test "x${doQReal}" = "xyes"; then
    omniQPrec=$qRealPrecBits
    AC_DEFINE(ENABLE_QREAL)
    AH_TEMPLATE(ENABLE_QREAL, [quad real])
    AC_DEFINE_UNQUOTED(OMNI_QREAL_PREC, $omniQPrec)
    AH_TEMPLATE(OMNI_QREAL_PREC, [quad real])

    qcCC () {
    rm -f $2 ${2}.*
    ${CC} -I. ${CPPFLAGS} ${CFLAGS} -DENABLE_QREAL ${hasLLFlags} ${gmpIncOpt} $3 $1 -o $2 ${gmpLibOpt}
    if test -x $2; then
        ret=`$2 $4`
        if test "x${ret}" = "x"; then
            AC_MSG_RESULT([can't determine??. exit.])
            exit 1
        else
            rm -f $2 ${2}.*
            echo ${ret}
        fi
    else
        AC_MSG_RESULT([can't create check program. exit.])
        exit 1
    fi
    }

    # check size of mp_exp_t
    AC_MSG_CHECKING([size of GNU-MP mp_exp_t])
    eSz=`qcCC buildutils/csMPexp.c ./csMPexp`
    AC_MSG_RESULT([${eSz}])
    omniExpT=""
    case ${eSz} in
    2)
        omniExpT=${TYPE_INT16};;
    4)
        omniExpT=${TYPE_INT32};;
    8)
        omniExpT=${TYPE_INT64};;
    *)
        AC_MSG_RESULT([can't determine basic type of mp_exp_t ??. exit.])
        exit 1;;
    esac
    AC_DEFINE_UNQUOTED(OMNI_SIZEOF_QEXP_T, $eSz)
    AH_TEMPLATE(OMNI_SIZEOF_QEXP_T, [size of mp_exp_t])

    # check size of mp_limb_t
    AC_MSG_CHECKING([size of GNU-MP mp_limb_t])
    lSz=`qcCC buildutils/csMPlimb.c ./csMPlimb`
    AC_MSG_RESULT([${lSz}])
    omniLimbT=""
    case ${lSz} in
    2)
        omniLimbT=${TYPE_INT16};;
    4)
        omniLimbT=${TYPE_INT32};;
    8)
        omniLimbT=${TYPE_INT64};;
    *)
        AC_MSG_RESULT([can't determine basic type of mp_limb_t ??. exit.])
        exit 1;;
    esac
    AC_DEFINE_UNQUOTED(OMNI_SIZEOF_QLIM_T, $lSz)
    AH_TEMPLATE(OMNI_SIZEOF_QLIM_T, [size of mp_limb_t])

    # check how many mp_limb_t the GNU-MP needed for N bit precision.
    AC_MSG_CHECKING([how many mp_limb_t needed for $omniQPrec bit precision])
    omniQRealPrecInLimb=`qcCC buildutils/cMPlb.c ./cMPlb '' $omniQPrec`
    AC_MSG_RESULT([$omniQRealPrecInLimb])
    AC_DEFINE_UNQUOTED(OMNI_QLIM_LEN, $omniQRealPrecInLimb)
    AH_TEMPLATE(OMNI_QLIM_LEN, [how many mp_limb_t the GNU-MP needed for N bit precision])

    # check _omQReal_t needs a gap between _mp_exp and _mp_d[omniQRealPrecInLimb].
    AC_MSG_CHECKING([_omQReal_t needs a gap between _mp_exp and _mp_d])
    gapInByte=`expr $lSz - $eSz`
    subStr=""
    if test ${gapInByte} -gt 0; then
        case ${gapInByte} in
        2)
            # lim == int, exp == short
            subStr="${TYPE_INT16} _gap_;";;
        4)
            # lim == long long, exp == int
            subStr="${TYPE_INT32} _gap_;";;
        *)
            # lim == long long, exp == short
            changequote(<<, >>)
            subStr="char _gap_[${gapInByte}];"
            changequote([, ])
            ;;
        esac
    fi
    if test "x${subStr}" != "x"; then
        #TODO
        AC_MSG_RESULT([yes, put a "${subStr}" member in _omQReal_t.])
        omniQRealFillGap=${subStr}
        AC_DEFINE(OMNI_QREAL_NEED_GAP_MEMBER)
        AH_TEMPLATE(OMNI_QREAL_NEED_GAP_MEMBER, [])
        AC_DEFINE_UNQUOTED(OMNI_QREAL_GAP_SIZE, ${gapInByte})
        AH_TEMPLATE(OMNI_QREAL_GAP_SIZE, [GNU-MP gapInByte])
    else
        omniQRealFillGap="/* if ((sizeof(mp_exp_t) < sizeof(mp_limb_t)), here must be a gap filling member. */"
        AC_MSG_RESULT([no])
    fi

    # check libgmp has an mpf_floor(), appeared in gmp3 later.
    oLIBS=${LIBS}
    LIBS="${LIBS} ${gmpLibOpt}"
    oCFLAGS=${CFLAGS}
    CFLAGS="${CFLAGS} ${CPPFLAGS} -DENABLE_QREAL ${gmpIncOpt}"
    AC_MSG_CHECKING([mpf_floor() in $LIBS])
    hasFloor=no
    AC_TRY_LINK(
    [
        #include "gmp.h"
    ],
    [
        mpf_t a;
        mpf_t b;
        (void)mpf_floor(a, b);
    ],
    [AC_MSG_RESULT(yes); hasFloor=yes],
    [AC_MSG_RESULT(no)]
    )
    LIBS=${oLIBS}
    CFLAGS=${oCFLAGS}
    if test "x${hasFloor}" = "xno"; then
        AC_MSG_RESULT([You have to use gmp3 or later. exit.])
        exit 1
    fi
fi

AC_SUBST(doQReal)
AC_SUBST(gmpIncOpt)
AC_SUBST(gmpLibOpt)
AC_SUBST(omniQPrec)
AC_SUBST(omniExpT)
AC_SUBST(omniLimbT)
AC_SUBST(omniQRealPrecInLimb)
AC_SUBST(omniQRealFillGap)

# --------------------------------------------------------------------
# specify libxmp path
default_libxmlDir="/usr"

AC_ARG_WITH(libxml,
    [  --with-libxml=DIR           libxml installation path (default: /usr)],
    [LIBXML_PREFIX="${withval}"], [LIBXML_PREFIX="${default_libxmlDir}"])

if ! test -f $LIBXML_PREFIX/include/libxml2/libxml/parser.h; then
   AC_MSG_ERROR([parser.h not found. Please attach or check --with-libxml=<DIR>. See ./configure --help])
fi

AC_SUBST(LIBXML_PREFIX)

# --------------------------------------------------------------------
# TCA
AC_ARG_WITH(TCA,
    [  --with-TCA=DIR              TCA installation path],
    [TCA_PREFIX="${withval}"])

use_tca=no
if test "${TCA_PREFIX}" != ""; then
   use_tca=yes
   if ! test -f $TCA_PREFIX/tca-api.h; then
      AC_MSG_ERROR([$TCA_PREFIX/tca-api.h not found. Please check --with-TCA=<DIR>])
   fi
fi

AM_CONDITIONAL(TCA, test x${use_tca} = xyes)
AC_SUBST(TCA_PREFIX)

# --------------------------------------------------------------------
# specify GASNet path

default_gasnetDir=""

AC_ARG_WITH(gasnet,
    [  --with-gasnet=DIR           GASNet installation path],
    [GASNET_PREFIX="${withval}"], [GASNET_PREFIX="${default_gasnetDir}"])

use_gasnet=no
if test "${GASNET_PREFIX}" != ""; then
  use_gasnet=yes
  if ! test -f "${GASNET_PREFIX}/include/gasnet.h"; then
	  AC_MSG_ERROR([${GASNET_PREFIX}/include/gasnet.h not found. Please check --with-gasnet])
  fi
fi

AM_CONDITIONAL(GASNET, test x${use_gasnet} = xyes)
AC_SUBST(GASNET_PREFIX)

# --------------------------------------------------------------------
# specify GASNet conduit

AC_ARG_WITH(gasnet_conduit,
    [  --with-gasnet-conduit=NAME  GASNet conduit],
    [GASNET_CONDUIT="${withval}"], [GASNET_CONDUIT="${default_gasnet_conduit}"])

if test "${use_gasnet}" = yes; then
  gasnet_conduit_flag=no
  if test "${GASNET_CONDUIT}" = ""; then
    for c in smp udp mpi gm lapi elan vapi ibv portals gemini shmem dcmf sci
    do
      if test -d ${GASNET_PREFIX}/include/${c}-conduit; then
         GASNET_CONDUIT=${c}
         gasnet_conduit_flag=yes
      fi
    done
  else
    gasnet_conduit_flag=yes
  fi

  if test "${gasnet_conduit_flag}" = no || ! test -d $GASNET_PREFIX/include/${GASNET_CONDUIT}-conduit; then
    AC_MSG_ERROR([${GASNET_PREFIX}/include/${GASNET_CONDUIT}-conduit not found. Please attach or check --with-gasnet-conduit])
  else
    # Create temporary Makefile to set $(GASNET_CPPFLAGS) $(GASNET_CFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS)
    __ECHO=""
    if test $OS = "darwin"; then
        __ECHO="echo"
    else
	__ECHO="echo -e"
    fi
    tmp_file=_XMP_makefile.$$
    mak_file=${GASNET_PREFIX}/include/${GASNET_CONDUIT}-conduit/${GASNET_CONDUIT}-seq.mak

    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_CPPFLAGS)" > ${tmp_file}
    GASNET_CPPFLAGS=`make -s -f ${tmp_file} 2>/dev/null`
    GASNET_CPPFLAGS="${GASNET_CPPFLAGS} -D_XMP_COARRAY_GASNET"
    
    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_CFLAGS)"   > ${tmp_file}
    GASNET_CFLAGS=`make -s -f ${tmp_file} 2>/dev/null`

    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_LDFLAGS)"  > ${tmp_file}
    GASNET_LDFLAGS=`make -s -f ${tmp_file} 2>/dev/null`

    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_LIBS)"     > ${tmp_file}
    GASNET_LIBS=`make -s -f ${tmp_file} 2>/dev/null`
    rm ${tmp_file}
    unset __ECHO
  fi
fi

AC_SUBST(GASNET_CONDUIT)
AC_SUBST(GASNET_CPPFLAGS)
AC_SUBST(GASNET_CFLAGS)
AC_SUBST(GASNET_LDFLAGS)
AC_SUBST(GASNET_LIBS)

# --------------------------------------------------------------------
# specify GPU C compiler

AC_ARG_WITH(gpucc,
    [  --with-gpucc=CMD            GPU C compiler (default: nvcc)],
    [GPU_CMD="${withval}"], [GPU_CMD="nvcc"])

AC_SUBST(GPU_CMD)

# --------------------------------------------------------------------
# specify GPU C compiler flags

AC_ARG_WITH(gpu-cflags,
    [  --with-gpu-cflags=CFLAGS    GPU C compiler flags (default: -arch=sm_20 -O3)],
    [GPU_CFLAGS="${withval}"], [GPU_CFLAGS="-arch=sm_20 -O3"])

AC_SUBST(GPU_CFLAGS)

# --------------------------------------------------------------------
# specify libgpu path

AC_ARG_WITH(cuda,
    [  --with-cuda=DIR             CUDA installation path (default: /usr/local/cuda)],
    [GPGPU_PREFIX="${withval}"], [GPGPU_PREFIX="/usr/local/cuda"])

AC_SUBST(GPGPU_PREFIX)

# --------------------------------------------------------------------
# Export Omni Config Variable

OMNI_HOME=${prefix}
AC_SUBST(OMNI_HOME)
AH_TEMPLATE(OMNI_HOME, [omni compiler install directory])
AC_DEFINE_UNQUOTED(OMNI_HOME, "${OMNI_HOME}")

# If "--sysconfdir=" is set,
if test "${sysconfdir}" != '${prefix}/etc'; then
  OM_DRIVER_CONF_DIR="${sysconfdir}"
else
  OM_DRIVER_CONF_DIR="${OMNI_HOME}/etc"
fi

AC_SUBST(OM_DRIVER_CONF_DIR)
AH_TEMPLATE(OM_DRIVER_CONF_DIR, [omni compiler driver configuration file directory])
AC_DEFINE_UNQUOTED(OM_DRIVER_CONF_DIR, "${OM_DRIVER_CONF_DIR}")

  if test "$target" = "powerpc-ibm-none"; then
    OM_EXEC_C_PP="gcc -E"
  else
    OM_EXEC_C_PP="${BECC} -E"
  fi


#OM_EXEC_C_PP="${BECC} -E"
AC_SUBST(OM_EXEC_C_PP)

OM_EXEC_C_COMPILER="${BECC}"
AC_SUBST(OM_EXEC_C_COMPILER)

OM_EXEC_C_LINKER="${BECC}"
AC_SUBST(OM_EXEC_C_LINKER)

OM_EXEC_CXX_LINKER="${CXX}"
AC_SUBST(OM_EXEC_CXX_LINKER)

OM_EXEC_F_PP="${FC} -E"
AC_SUBST(OM_EXEC_F_PP)

OM_EXEC_F_COMPILER="${FC}"
AC_SUBST(OM_EXEC_F_COMPILER)

OM_EXEC_F_LINKER="${FC}"
AC_SUBST(OM_EXEC_F_LINKER)

INCLUDECOM=include
AC_SUBST(INCLUDECOM)

XMPPKG_CONFFILE="./release/package_param.conf"

if test -f "${XMPPKG_CONFFILE}"; then
	. "${XMPPKG_CONFFILE}"
else
	AC_MSG_WARN([a configuration file for packaging is not found.])
fi

XMPPKG_CHANGELOG_DATE=`date '+%a, %d %b %Y %H:%M:%S %z'`

AC_SUBST(XMPPKG_NAME)
AC_SUBST(XMPPKG_VERSION)
AC_SUBST(XMPPKG_SUMMARY)
AC_SUBST(XMPPKG_LICENSE)
AC_SUBST(XMPPKG_RELEASE)
AC_SUBST(XMPPKG_MAINTAINER)
AC_SUBST(XMPPKG_PREFIX)
AC_SUBST(XMPPKG_DESCRIPTION)
AC_SUBST(XMPPKG_URL)
AC_SUBST(XMPPKG_GROUP)
AC_SUBST(XMPPKG_PACKAGER)
AC_SUBST(XMPPKG_VENDOR)
AC_SUBST(XMPPKG_SECTION)
AC_SUBST(XMPPKG_PRIORITY)
AC_SUBST(XMPPKG_CHANGELOG_DATE)
AM_SUBST_NOTMAKE(XMPPKG_DESCRIPTION)

AC_CONFIG_FILES(
	Makefile
	release/debian/changelog
	release/debian/control
	release/debian/rules
	release/xcalablemp.spec
	release/make_rpm.sh
	release/make_deb.sh
)

# --------------------------------------------------------------------
ATOOL_DIR=./Driver/atool
AC_SUBST(ATOOL_DIR)

RM="rm -f"
AC_SUBST(RM)

MKDIR="mkdir -p"
AC_SUBST(MKDIR)

# headers/macros
AC_CHECK_HEADER(sys/param.h, [AC_DEFINE([HAVE_SYS_PARAM_H], [1], [ ])])
AC_CHECK_HEADER(ctype.h, [AC_DEFINE([HAVE_CTYPE_H], [1], [ ])])
AC_CHECK_HEADER(stdio.h, [AC_DEFINE([HAVE_STDIO_H], [1], [ ])])
AC_CHECK_HEADER(stdarg.h, [AC_DEFINE([HAVE_STDARG_H], [1], [ ])])
AC_CHECK_HEADER(dirent.h, [AC_DEFINE([HAVE_DIRENT_H], [1], [ ])])
AC_CHECK_HEADER(time.h, [AC_DEFINE([HAVE_TIME_H], [1], [ ])])
AC_CHECK_HEADER(sys/termios.h, [AC_DEFINE([HAVE_SYS_TERMIOS_H], [1], [ ])])
AC_CHECK_HEADER(signal.h, [AC_DEFINE([HAVE_SIGNAL_H], [1], [ ])])
AC_CHECK_HEADER(netdb.h, [AC_DEFINE([HAVE_NETDB_H], [1], [ ])])
AC_CHECK_HEADER(netinet/tcp.h, [AC_DEFINE([HAVE_NETINET_TCP_H], [1], [ ])])
AC_CHECK_HEADER(sys/socket.h, [AC_DEFINE([HAVE_SYS_SOCKET_H], [1], [ ])])
AC_CHECK_HEADER(arpa/inet.h, [AC_DEFINE([HAVE_ARPA_INET_H], [1], [ ])])
AC_CHECK_HEADER(sys/ioctl.h, [AC_DEFINE([HAVE_SYS_IOCTL_H], [1], [ ])])
AC_CHECK_HEADER(pthread.h, [AC_DEFINE([HAVE_PTHREAD_H], [1], [ ])])
AC_CHECK_HEADER(stdbool.h, [AC_DEFINE([HAVE_STDBOOL_H], [1], [ ])])
AC_CHECK_HEADER(sys/resource.h, [AC_DEFINE([HAVE_SYS_RESOURCE_H], [1], [ ])])
AC_CHECK_HEADER(regex.h, [AC_DEFINE([HAVE_REGEX_H], [1], [ ])])
AC_CHECK_FUNC(strtold, [AC_DEFINE([HAVE_STRTOLD], [1], [ ])])

# Assume 32 bit for cross compilation.
AC_CHECK_SIZEOF(short int, 2)
AC_CHECK_SIZEOF(int, 4)
AC_CHECK_SIZEOF(long int, 4)
AC_CHECK_SIZEOF(long long, 8)
AC_CHECK_SIZEOF(int64_t, 8)
AC_CHECK_SIZEOF(size_t, 4)
AC_CHECK_SIZEOF(void *, 4)

oCPPFLAGS=${CPPFLAGS}
oCFLAGS=${CFLAGS}
CPPFLAGS=''
CFLAGS="${CFLAGS} -Wall -Werror"
AC_MSG_CHECKING([print format for size_t])
AC_RUN_IFELSE(
	[AC_LANG_PROGRAM(
		[[#include <stdio.h>]
		 [#include <stddef.h>]
		 [#include <string.h>]],
		[[char buf[24];]
		 [snprintf(buf, sizeof(buf), "%zu", (size_t)1);]
		 [return (strcmp("1", buf) == 0 && strlen(buf) == 1) ? ]
		 [0 : 1;]])],
	[AC_DEFINE([HAVE_PRINT_FORMAT_FOR_SIZE_T], [], [ ])
	 AC_MSG_RESULT([available.])],
	[AC_MSG_RESULT([not available.])],
	[AC_MSG_RESULT([not checkable since cross-compilation.])])
CFLAGS=${oCFLAGS}
CPPFLAGS=${oCPPFLAGS}

# where the omni is.
tmpOmniDir=''
omniDir="${OMNI_HOME}"
omniBinDir=''
omniLibDir=''
omniEtcDir="${OM_DRIVER_CONF_DIR}/"
omniIncDir=''
AC_ARG_WITH([omni],
	[AS_HELP_STRING([--with-omni],
		[specify where the omni is installed (prefix default).])],
	[tmpOmniDir=${withval}],
	[tmpOmniDir=${prefix}])
if test -z "${tmpOmniDir}"; then
    tmpOmniDir=${defaultOmniDir}
fi
AC_MSG_CHECKING([the Omni is installed at "${tmpOmniDir}"])
if test -d "${tmpOmniDir}"; then
    if test -x "${tmpOmniDir}/bin/F_Front"; then
	omniDir="${tmpOmniDir}"
	AC_MSG_RESULT([yes.])
    else
	omniDir=''
	AC_MSG_RESULT([no. Assume every user sets path correctly.])
    fi
fi
if test ! -z "${omniDir}"; then
    # Don't delete the last slash.
    omniBinDir="${omniDir}"/bin/
    omniLibDir="${omniDir}"/lib/
#    omniEtcDir="${omniDir}"/etc/
    omniIncDir="${omniDir}"/include
fi

AC_SUBST(omniDir)
AC_SUBST(omniBinDir)
AC_SUBST(omniLibDir)
AC_SUBST(omniEtcDir)
AC_SUBST(omniIncDir)

# native compiler
tmpNativeCompiler=''
AC_ARG_WITH([nativeCompiler],
	[AS_HELP_STRING([--with-nativeCompiler],
		[specify a native fortran compiler ("gfortran" default).])],
	[tmpNativeCompiler=${withval}],
	[tmpNativeCompiler="gfortran"])
nativeCompiler=${tmpNativeCompiler}
AC_SUBST(nativeCompiler)

checkCpp() {
    __ret=1
    __tmpFileBase=/tmp/.cpptest.$$
    __tmpFile=${__tmpFileBase}.F90
    __data=`date | base64 2>/dev/null`
    echo "#ifdef __XMP_CPP_CHECK__" > ${__tmpFile}
    echo ${__data} >> ${__tmpFile}
    echo "#endif" >> ${__tmpFile}
    __vdata=`$* -D__XMP_CPP_CHECK__ ${__tmpFile} 2>/dev/null | grep "${__data}" 2>/dev/null`
    rm -f ${__tmpFileBase}*
    if test "x${__data}" = "x${__vdata}"; then
        __ret=0
    fi
    unset __tmpFile __data __vdata
    return ${__ret}
}

# cpp
cppCmd=''
tmpCppCmd=''
AC_ARG_WITH([cpp],
	[AS_HELP_STRING([--with-cpp],
		[specify a cpp command (native compiler default).])],
	[tmpCppCmd=${withval}],
	[tmpCppCmd=${nativeCompiler}])
AC_MSG_CHECKING(["${tmpCppCmd}" is a C preprosessor])
checkCpp ${tmpCppCmd}
if test $? -eq 0; then
    AC_MSG_RESULT([yes, it is.])
    cppCmd=${tmpCppCmd}
else
    AC_MSG_RESULT([no, it isn't.])
    AC_MSG_CHECKING(["${nativeCompiler} -E" acts like cpp])
    checkCpp ${nativeCompiler} -E
    if test $? -eq 0; then
        AC_MSG_RESULT([yes, it does.])
        cppCmd=${nativeCompiler}
    else
	AC_MSG_RESULT([no, it doesn't.])
    fi
fi
if test -z "${cppCmd}"; then
   AC_MSG_FAILURE([Can't find any usable C preprosessor.])
   exit 1
fi
AC_SUBST(cppCmd)

# sysdeps.

#eval `echo ${target} | awk -F- '{ printf "tagCPU=%s; tagOS=%s;\n", $1, $3 }'`
#changequote(<<, >>)
#CPU=`echo ${tagCPU} | tr '[A-Z]' '[a-z]'`
#OS=`echo ${tagOS} | tr '[A-Z]' '[a-z]'` 
#changequote([, ])
#unset tagOS tagCPU

HOSTARCH=`./buildutils/config.guess | grep -v Linking`
CPU=`echo $HOSTARCH | awk -F- '{ print $1 }' | sed 's/\./_/g'`
OS=`echo $HOSTARCH | awk -F- '{ print $3 }' | sed 's/\./_/g'`


case $CPU in
    i*86)
        CPU=i386;;
esac

OS_CPPFLAGS='-D_REENTRANT'
OS_LDFLAGS=''
OS_LIBS=''
OS_CFLAGS=''
if test -z "${MAKE}"; then
	MAKE=make
fi

API_FLAVOR=NATA_API_POSIX
TARGET_ARCH=POSIX
case $OS in
    sunos*)
        OS=sunos;;
    solaris*)
        OS=solaris;;
    aix*)
	MAKE=gmake;; 
    freebsd*)
	OS=freebsd;;
    linux*)
	OS_CPPFLAGS="${OS_CPPFLAGS} -D_GNU_SOURCE"
	OS_LIBS="-lrt -lpthread"
	OS=linux;;
    netbsd*)
        MAKE=gmake
        OS_CPPFLAGS="${OS_CPPFLAGS} -I/usr/pkg/include"
    	OS_LDFLAGS="-L/usr/pkg/lib"
	OS_LIBS="-lpthread"
	OS=netbsd;;
    cygwin*)
	OS=cygwin;;
    mingw32ce)
	OS=windowsce
	TARGET_ARCH=WINDOWS
	API_FLAVOR=NATA_API_WIN32API;;
    mingw32*)
	case $CPU in
	    x86_64*)
		OS=windows64;;
	    i*86)
		OS=windows32;;
	    *)
		OS=windows;;
	esac
	OS_LIBS="-lpthread -loleaut32 -lws2_32 -lwinmm"
	TARGET_ARCH=WINDOWS
	API_FLAVOR=NATA_API_WIN32API;;
esac

if test "$target" != "$host"; then
   case "$target" in
        Kcomputer-linux-gnu|FX10-linux-gnu)
                LDFLAGS="";;
        MIC-linux-gnu)
                LDFLAGS="";;
        sx-nec-superux)
                LDFLAGS="-L/usr/lib64";;
        powerpc64-hitachi-aix)
                LDFLAGS="";;
        powerpc-ibm-none)
                LDFLAGS="";;
        *)
                LDFLAGS="";;
   esac

else

   CC_LIB=${CC}
   AC_SUBST(CC_LIB)

fi

OS_CXXFLAGS=${OS_CFLAGS}
AC_SUBST(MAKE)
AC_SUBST(OS_CPPFLAGS)
AC_SUBST(OS_CFLAGS)
AC_SUBST(OS_CXXFLAGS)
AC_SUBST(OS_LDFLAGS)
AC_SUBST(OS_LIBS)

changequote(<<, >>)
CPUDEF=NATA_CPU_`echo $CPU | tr '[a-z]' '[A-Z]'`
OSDEF=NATA_OS_`echo $OS | tr '[a-z]' '[A-Z]'`
changequote([, ])
AC_SUBST(CPUDEF)
AC_SUBST(OSDEF)
AC_SUBST(API_FLAVOR)
AC_SUBST(TARGET_ARCH)

MKRULESDIR=${TOPDIR}/Driver/atool/mk
AC_SUBST(MKRULESDIR)

AC_SUBST(CPPFLAGS)
AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)
AC_SUBST(LDFLAGS)

# emit.
AC_CONFIG_FILES(
	Driver/atool/mk/vars.mk
	Driver/atool/Makefile
	Driver/atool/src/lib/Makefile
	Driver/atool/src/include/nata/Makefile
	Driver/atool/src/include/nata/nata_platform.h
	Driver/atool/src/cmd/Makefile
	Driver/atool/src/cmd/paths.h
)

AC_CONFIG_COMMANDS([modify-config],
	[(cd Driver/atool && ${BASH} ./mk/finalizeconfig ${OS_CPPFLAGS})],
	[OS_CPPFLAGS="${OS_CPPFLAGS}"])

AC_CONFIG_COMMANDS([includes],
	[${BASH} ./Driver/atool/mk/initheaders])

AC_CONFIG_COMMANDS([depend],
	[(${BASH} ./Driver/atool/mk/initdepend && cd ./Driver/atool && ${MAKE} depend)],
	[MAKE=${MAKE}])

AC_CONFIG_COMMANDS([config-clean],
	[rm -rf ./config.log ./config.status.* autom4te.cache])

unset program_transform_name

AC_OUTPUT
${MAKE} clean

# ----------------------
# Edit version file
VERSIONTXT="./Driver/etc/version"
if test -d .git; then
  VERSION=`git log --date=short --pretty=format:"Last Commit Day:%ad, Hash Value:%h" -1`
  echo -n Version:${XMPPKG_VERSION}, ${VERSION} > ${VERSIONTXT}
fi

# ----------------------
# Output

cat << EOF
==================================================
  -- Configuration Summary --

  PREFIX : ${prefix}

  Local Compiler:
    C Compiler            : $CC
    C Compiler FLAG       : "$CFLAG"
    Fortran Compiler      : $FC
    Fortran Compiler FLAG : "$FFLAG"
  
  Backend Compiler:
    C Compiler            : $BECC
    C Compiler FLAG       : "$BECC_OPTION"
    Fortran Compiler      : $BEFC
    Fortran Compiler FLAG : "$BEFC_OPTION"
EOF

if test "$target" != "Kcomputer-linux-gnu" -a "$target" != "FX10-linux-gnu"; then
  echo ""
  echo "  Options:"
  echo "    GASNet                : $use_gasnet"

  if test "${use_gasnet}" = yes; then
  cat << EOF
      Installed in        : $GASNET_PREFIX
      Conduit             : $GASNET_CONDUIT

EOF
  fi
  echo "    OpenACC               : $doOpenAcc"
fi

if test "${use_tca}" = "yes"; then
  echo "    TCA                   : yes"
fi
echo "=================================================="

