# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
abspath()
{
    cd $1 > /dev/null
    if [ $? != 0 ]; then
        echo file not found : $1
    fi
    echo $PWD
    cd - > /dev/null
}

work=`abspath ../..`
export OMNI_HOME=$work
frontend=$work/F-FrontEnd/src/F_Front
backend=$work/F-BackEnd/bin/F_Back
nativecomp=gfortran
tmpdir=$work/compile
testdata=$work/F-FrontEnd/test/testdata

chmod +x $backend
if [ ! -e "$tmpdir" ]; then
    mkdir $tmpdir
fi

cd $tmpdir
if [ $? != 0 ]; then
    exit 1
fi

ulimit -t 10

echo > errors.txt

for f in $testdata/*.f $testdata/*.f90; do
    b=`basename $f`
    $frontend -fopenmp -I $testdata $f -o $b.xml > $b.out 2>&1
    if [ $? = 0 ]; then
        $backend $b.xml -o $b.dec.f90 >> $b.out 2>&1
        if [ $? = 0 ]; then
            $nativecomp -c $b.dec.f90 -o $b.o >> $b.out 2>&1
            if [ $? = 0 ]; then
                echo ok: $b
            else
                echo --- failed native: $b | tee -a errors.txt
            fi
        else
            echo --- failed backend: $b | tee -a errors.txt
        fi
    else
        echo --- failed frontend: $b | tee -a errors.txt
    fi
done

