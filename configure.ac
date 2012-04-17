dnl    This file is an input file used by the GNU "autoconf" program to
dnl    generate the file "configure", which is run during Omni installation
dnl    to configure the system for the local environment.

AC_PREREQ(2.60)

# --------------------------------------------------------------------
# Let the initiation begin....

AC_INIT([Omni XcalableMP Compiler], [1], [BUG-REPORT-ADDRESS], [omni_xmp])

AC_CONFIG_AUX_DIR([buildutils])
AC_CONFIG_MACRO_DIR([buildutils])

AC_CANONICAL_TARGET

AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_SRCDIR([include/exc_platform.h])
AC_CONFIG_HEADER([include/config.h])
AC_PREFIX_DEFAULT(/usr/local/omni3)

if test "${prefix}" = "NONE"; then
	prefix=/usr/local/omni3
fi

# --------------------------------------------------------------------

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

TESTS_DIR=tests
AC_CONFIG_FILES($TESTS_DIR/C-test/Makefile)
AC_CONFIG_FILES($TESTS_DIR/tiny/Makefile)
AC_CONFIG_FILES($TESTS_DIR/clinpack/Makefile)
AC_CONFIG_FILES($TESTS_DIR/F-test/OMP-test/Makefile)

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
    [  --with-thread           type of openmp thread],
    [thread_type=$withval], [thread_type=""])

# Default lock type
# Check whether --with-lock or --without-lock was given.
AC_ARG_WITH(lock,
    [  --with-lock             type of openmp lock],
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
LIBTLOG_DIR="libtlog/src_threads"
AC_CONFIG_FILES($LIBTLOG_DIR/Makefile)
LIBTLOG_MPI_DIR="libtlog/src_mpi"
AC_CONFIG_FILES($LIBTLOG_MPI_DIR/Makefile)
AC_CONFIG_FILES($LIBTLOG_MPI_DIR/bin/tlogview)
genLibTargetTmpl $LIBOMPC_DIR

AC_SUBST(LIBOMPC_DIR)
AC_SUBST(LIBTLOG_DIR)
AC_SUBST(LIBTLOG_MPI_DIR)

# --------------------------------------------------------------------
# check what CC will be used, and set compile flags

isGcc=yes

optCFlags=""

debCFlags=""
userCFlags=""
mustCFlags=""

debCppFlags=""
userCppFlags=""
mustCppFlags=""

ccCom=""
AC_ARG_WITH(cflag,
    [  --with-cflag=OPT        specify C compiler options for optimization],
    [userCFlags="${withval}"], [userCFlags=""])

AC_ARG_ENABLE(debug,
    [  --enable-debug          enable generate executable with debug symbol (false default).],
    [doDebug=$enableval], [doDebug=no])

AC_ARG_ENABLE(xmpgpu,
    [  --enable-xmpgpu         enable XcalableMP-GPU (false default).],
    [doXmpGpu=$enableval], [doXmpGpu=no])

AC_ARG_WITH(cppflag,
    [  --with-cppflag=OPT      specify C preprocessor options],
    [userCppFlags="${withval}"], [userCppFlags=""])

AC_PROG_CC
AC_PROG_CXX

if test "x${doDebug}" = "xyes"; then
    debCFlags="-g"
    optCFlags=""
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

# check gcc version
AC_MSG_CHECKING(gcc version)
changequote(<<, >>)dnl
gccVer=`${CC} --version 2>&1 | head -n1 | sed 's:[^ ]* [^ ]* ::' | sed 's: .*::'`
changequote([, ])dnl
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

CPPFLAGS="${CPUDEF} ${OSDEF} ${debCppFlags} ${userCppFlags} ${mustCppFlags}"
AC_SUBST(CPPFLAGS)

CFLAGS="${debCFlags} ${userCFlags} ${optCFlags} ${mustCFlags} ${CFLAGS}"
CXXFLAGS=$CFLAGS
AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)

AH_TEMPLATE(OBJEXT, [object file extension])
AC_DEFINE_UNQUOTED(OBJEXT, "${OBJEXT}")


if test "$BECC" = ""; then
	BECC=${CC}
fi

AC_MSG_CHECKING([backend C compiler])

chk=`$BECC --version | head -n1`
case $chk in
gcc*)
    OMC_CONF_LX2X_OPT=""
    OMC_CONF_NTV_OPT="-std=gnu99 -Wno-implicit"
    OMC_CONF_LNK_OPT=""
    ;;
*ICC*)
    OMC_CONF_LX2X_OPT=""
    OMC_CONF_NTV_OPT="-std=c99 -w"
    OMC_CONF_LNK_OPT=""
    ;;
*)
    echo "No supported C compiler was found."
    echo "This program supports following C compilers."
    echo " - GNU C Compiler (gcc)"
    echo " - Intel C Compiler (icc)"
    exit 1
    ;;
esac

AC_SUBST(OMC_CONF_LX2X_OPT)
AC_SUBST(OMC_CONF_NTV_OPT)
AC_SUBST(OMC_CONF_LNK_OPT)
AC_SUBST(BECC)

AC_MSG_RESULT([$BECC])

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# check Fortran Compiler

if test $enableF2F = yes; then

    AC_PROG_FC([${FC},gfortran,ifort])

    AC_MSG_CHECKING([backend Fortran compiler])
    if test "$BEFC" = ""; then
        BEFC=${FC}
    fi

    if test "$BEFC" != ""; then
        chk=`$BEFC --version | head -n1`
        case $chk in
        GNU*)
            OMF_CONF_LX2X_OPT="-gnu"
            OMF_CONF_NTV_OPT="-ffree-form"
            OMF_CONF_LNK_OPT=""
            ;;
        *IFORT*)
            OMF_CONF_LX2X_OPT="-intel"
            OMF_CONF_NTV_OPT="-auto"
            OMF_CONF_LNK_OPT="-threads"
            ;;
        *)
            echo "No supported Fortran compiler was found."
            echo "This program supports following Fortran compilers."
            echo " - GNU Fortran (gfortran)"
            echo " - Intel Fortran (ifort)"
            exit 1
            ;;
        esac
        AC_SUBST(OMF_CONF_LX2X_OPT)
        AC_SUBST(OMF_CONF_NTV_OPT)
        AC_SUBST(OMF_CONF_LNK_OPT)
        AC_SUBST(BEFC)
        AC_MSG_RESULT([$BEFC])
    else
        echo "No Fortran compiler was found."
        exit 1
    fi

    AC_PROG_FC_C_O
fi

# --------------------------------------------------------------------

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

# check ranlib & ar
if test "$target" != "$host"; then

   case "$target" in
	Kcomputer-linux-gnu)
		AC_CHECK_TOOL([RANLIB], [ranlibpx], [:])
		AC_CHECK_TOOL([AR], [arpx], [:])
		ARFLAGS="cru"
		AC_SUBST(ARFLAGS) ;;
	*)
		AC_CHECK_TARGET_TOOL([RANLIB], [ranlib], [:])
		AC_CHECK_TARGET_TOOL([AR], [ar], [:]) ;;
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

fi

# check CC for libraries

if test "$target" != "$host"; then

   case "$target" in
	Kcomputer-linux-gnu)
		CC_LIB="fccpx -Xg -std=gnu99"
		AC_SUBST(CC_LIB) ;;
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
    AC_MSG_CHECKING([IBM Java6])
    jTmp=/tmp/.omniJavaChk.$$
    ${JAVACOM} -fullversion > ${jTmp} 2>&1
    chk=`grep 'J2RE 1.6.0 IBM' ${jTmp}`
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
    AC_MSG_CHECKING([Sun Java6])

    changequote(<<, >>)dnl
    chk=`${JAVACOM} -fullversion 2>&1 | sed -e 's:java[\t ]*full[\t ]*version[\t ]*::g' -e 's:\"::g'`
    changequote([, ])dnl

    if test "x${chk}" = "x"; then
        AC_MSG_RESULT([no])
    fi
    case $chk in
    1.6*|1.7*)
        isSunJava6=yes
        AC_MSG_RESULT([yes]);;
    *)
        isSunJava6=no
        AC_MSG_RESULT([no]);;
    esac
    unset chk
fi

if test "${isSunJava6}" = "no" -a "${isIBMJava6}" = "no"; then
    AC_MSG_ERROR([Java6 not found]);
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
javadir='${prefix}/java'
AC_SUBST(javadir)

# java option
JAVA_OPT="-Xmx1024m"
AC_SUBST(JAVA_OPT)


# --------------------------------------------------------------------

#--------------------------------------------------------------------
#    endian check

AC_C_BIGENDIAN

#--------------------------------------------------------------------

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

# --------------------------------------------------------------------
# Checks for header files

# check stdlib.h, stdarg.h, string.h, float.h
AC_HEADER_STDC
# check if needed sys/time.h with time.h
AC_HEADER_TIME

AC_CHECK_HEADERS([fcntl.h libintl.h malloc.h stddef.h stdint.h limits.h strings.h unistd.h errno.h locale.h sys/time.h sys/resource.h sys/wait.h sys/stat.h assert.h])

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Checks for library functions.

AC_FUNC_MALLOC
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

# --------------------------------------------------------------------

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
AC_CHECK_FUNCS(gettimeofday, , AC_DEFINE(NO_GETTOD))
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


# --------------------------------------------------------------------
# quad real support

if test "${OS}" = "linux"; then
    gmpIncDir=/usr/include
    gmpLibDir=/usr/lib
else
    gmpIncDir=/usr/local/include
    gmpLibDir=/usr/local/lib
fi

gmpIncOpt=""
gmpDirOpt=""
qRealPrecBits=128

AC_ARG_ENABLE(mreal,
    [  --enable-mreal          enable multiple precision real (and real*16) support of Fortran 77 (false default).],
    [doQReal=$enableval], [doQReal=no])

AC_ARG_WITH(gmpLibDir,
    [  --with-gmpLibDir=DIR    specify where the GNU-MP library is for quad-real support.],
    [gmpLibDir=$withval], [true])

AC_ARG_WITH(gmpIncDir,
    [  --with-gmpIncDir=DIR    specify where the GNU-MP include headers are for quad-real support.],
    [gmpIncDir=$withval], [true])

if test "x${doQReal}" = "xyes"; then
    if test -f ${gmpIncDir}/gmp.h; then
        true
    else
        AC_MSG_RESULT([can't find ${gmpIncDir}/gmp.h. exit.])
        exit 1
    fi
    gmpLibs=`ls ${gmpLibDir}/libgmp.* 2> /dev/null`
    if test "x${gmpLibs}" = "x"; then
        AC_MSG_RESULT([can't find ${gmpLibDir}/libgmp.*. exit.])
        exit 1
    fi
    gmpIncOpt="-I${gmpIncDir}"
    gmpLibOpt="-L${gmpLibDir} -lgmp"
fi

AC_ARG_WITH(precision,
    [  --with-precision=NUM    specify the default precision of real*16 in bits (128).],
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
    lSz=`qcCC utils/csMPlimb.c ./csMPlimb`
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
    omniQRealPrecInLimb=`qcCC utils/cMPlb.c ./cMPlb '' $omniQPrec`
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
            changequote(<<, >>)dnl
            subStr="char _gap_[${gapInByte}];"
            changequote([, ])dnl
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

if test "$target" == "Kcomputer-linux-gnu"; then
   default_libxmlDir="/usr"
   default_mpicc="mpifccpx -Xg -std=gnu99"
else
   default_libxmlDir="/usr/local"
   default_mpicc="mpicc"
fi

# --------------------------------------------------------------------
# specify libxmp path

AC_ARG_WITH(libxmlDir,
    [  --with-libxmlDir=DIR    specify libxml install directory (default: /usr/local)],
    [LIBXML_PREFIX="${withval}"], [LIBXML_PREFIX="${default_libxmlDir}"])

AC_SUBST(LIBXML_PREFIX)

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# specify MPI C compiler

AC_ARG_WITH(mpicc,
    [  --with-mpicc=CMD        specify MPI C compiler (default: mpicc)],
    [MPI_CMD="${withval}"], [MPI_CMD="${default_mpicc}"])

AC_SUBST(MPI_CMD)

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# specify GPU C compiler

AC_ARG_WITH(gpucc,
    [  --with-gpucc=CMD        specify GPU C compiler (default: nvcc)],
    [GPU_CMD="${withval}"], [GPU_CMD="nvcc"])

AC_SUBST(GPU_CMD)

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# specify libgpu path

AC_ARG_WITH(gpuDir,
    [  --with-gpuDir=DIR    specify GPGPU tooklit install directory (default: /usr/local/cuda)],
    [GPGPU_PREFIX="${withval}"], [GPGPU_PREFIX="/usr/local/cuda"])

AC_SUBST(GPGPU_PREFIX)

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Export Omni Config Variable

OMNI_HOME=$prefix
AC_SUBST(OMNI_HOME)
AH_TEMPLATE(OMNI_HOME, [omni compiler install directory])
AC_DEFINE_UNQUOTED(OMNI_HOME, "${OMNI_HOME}")

OM_DRIVER_CONF_DIR="${OMNI_HOME}/etc"
AC_SUBST(OM_DRIVER_CONF_DIR)
AH_TEMPLATE(OM_DRIVER_CONF_DIR, [omni compiler driver configuration file directory])
AC_DEFINE_UNQUOTED(OM_DRIVER_CONF_DIR, "${OM_DRIVER_CONF_DIR}")

OM_EXEC_C_PP="${BECC} -E"
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

# --------------------------------------------------------------------

# --------------------------------------------------------------------

AC_CONFIG_FILES(Makefile)

AC_OUTPUT



