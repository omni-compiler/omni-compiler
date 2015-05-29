BASE_TESTDIR=$1
shift
TESTDIRS=$@

for subdir in ${TESTDIRS}; do
    LINE=`find ${BASE_TESTDIR}$subdir | wc -l`
    echo $LINE $subdir
done | sort -rn
