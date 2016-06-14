#!/bin/sh

abspath() {
    __cwd=`pwd`
    cd $1 > /dev/null 2>&1
    if test $? -eq 0; then
	/bin/pwd
    fi
    cd ${__cwd}
    unset __cwd
}

work=`abspath ../..`
OMNI_SOURCE_DIR=${work}
export OMNI_SOURCE_DIR
setupConf="${work}/Driver/etc/setup.conf"
if test -r "${setupConf}"; then
    . "${setupConf}"
else
    echo "can't read '${setupConf}'."
    exit 1
fi

omniSetup
if test $? -ne 0; then
    error "can't initialize compiler environment."
    exit 1
fi

front="${_F_Front}"
trans="${_F_Trans}"
decomp="${_F_Decomp}"
nativecomp=gfortran
tmpdir=${work}/compile
testdata=${work}/F-FrontEnd/test/testdata

if test ! -e "${tmpdir}"; then
    mkdir -p "${tmpdir}"
fi

cd "${tmpdir}" || exit 1

useTrans=0
if test $# -gt 0; then
    useTrans=1
fi

ulimit -t 10

echo > errors.txt

for f in $testdata/*.f $testdata/*.f90; do

    b=`basename $f`
    errOut=${b}.out
    frontXMLOut=${b}.front.xml
    decompiledSrc=${b}.dec.f90
    binOut=${b}.o
    fOpts=''
    if test -f ${f}.options; then
	fOpts=`cat ${f}.options`
    fi

    ${front} ${F_FRONT_TEST_OPTS} ${fOpts} -I ${testdata} ${f} \
	-o ${frontXMLOut} > ${errOut} 2>&1
    if test $? -eq 0; then
	if test ${useTrans} -eq 0; then

            ${decomp} ${frontXMLOut} -o ${decompiledSrc} >> ${errOut} 2>&1
	    if test $? -eq 0; then
		${nativecomp} -c ${decompiledSrc} -o ${binOut} >> \
		    ${errOut} 2>&1
		if test $? -eq 0; then
                    echo ok: ${b}
		else
                    echo --- failed native: ${b} | tee -a errors.txt
		fi
            else
		echo --- failed backend: ${b} | tee -a errors.txt
            fi

	else

	    transXMLOut=${b}.trans.xml
	    ${trans} ${frontXMLOut} -o ${transXMLOut} >> ${errOut} 2>&1
	    if test $? -eq 0; then
		${decomp} ${transXMLOut} -o ${decompiledSrc} >> ${errOut} 2>&1
		if test $? -eq 0; then
		    ${nativecomp} -c ${decompiledSrc} -o ${binOut} >> \
			${errOut} 2>&1
		    if test $? -eq 0; then
			echo ok: ${b}
		    else
			echo --- failed native: ${b} | tee -a errors.txt
		    fi
		else
		    echo --- failed decompile: ${b} | tee -a errors.txt
		fi
	    else
		echo --- failed translate: ${b} | tee -a errors.txt
	    fi

	fi
    else
	echo --- failed frontend: ${b} | tee -a errors.txt
    fi

done

exit 0
