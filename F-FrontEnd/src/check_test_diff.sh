#!/bin/bash

# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $

RESULT_DIR="check_result"
BASE_DIR="$RESULT_DIR/base"
OUTPUT_DIR="$RESULT_DIR/now"

make || exit 1

ulimit -c 0 # I want no core dump

TMPFILE=$(mktemp)
trap "rm $TMPFILE" 0

should_be_ok="../test/testdata"
should_be_ng="../test/failtestdata"
result_no_care="../test/notsupported"

do_compile()
{
	local testfile="$1"

	local filename=$OUTPUT_DIR/$(md5sum $testfile | awk '{print $1}')
	./F_Front $testfile > $TMPFILE 2>/dev/null
	local exitcode=$?
	python replace_type.py $TMPFILE > $filename
	echo "END_OF_XCODEML" >> $filename
	echo "exit code: $exitcode" >> $filename
	if [ $exitcode -gt 1 ]; then
		# F_Front is segfault or abort, maybe.
		echo "Error: $testfile compile error" >&2
	fi
	return $exitcode
}

do_test_should_be_ok()
{
	local testfile
	for testfile in $(find $should_be_ok -name "*.f90" -o -name "*.f"); do
		do_compile $testfile
		if [ $? -ne 0 ]; then
			echo "Error: $testfile should have exit code == 0"
		fi
	done
}

do_test_should_be_ng()
{
	local testfile
	for testfile in $(find $should_be_ng -name "*.f90" -o -name "*.f"); do
		do_compile $testfile
		if [ $? -eq 0 ]; then
			echo "Error: $testfile should have exit code != 0"
		fi
	done
}

do_test_result_no_care()
{
	local testfile
	for testfile in $(find $result_no_care -name "*.f90" -o -name "*.f"); do
		do_compile $testfile
	done
}

show_diff()
{
	local have_diff=0
	local testfile

	for testfile in $(find $should_be_ok $should_be_ng $result_no_care -name "*.f90" -o -name "*.f") ; do
		md5=$(md5sum $testfile | awk '{print $1}')
		if [ ! -e $BASE_DIR/$md5 ]; then
			echo $testfile is modified or new
			continue;
		fi
		cmp $BASE_DIR/$md5 $OUTPUT_DIR/$md5 > /dev/null 2>&1
		if [ $? -ne 0 ]; then
			echo $testfile output differ
			diff -u $BASE_DIR/$md5 $OUTPUT_DIR/$md5
			have_diff=1
		fi
	done
	if [ $have_diff -eq 0 ]; then
		echo "No difference found in XcodeML outputs"
	fi
	echo To save this result for comparison base, plz execute
	echo "\$ rm -rf $BASE_DIR; mv $OUTPUT_DIR $BASE_DIR"
}

main()
{
	rm -rf $OUTPUT_DIR
	mkdir -p $OUTPUT_DIR

	do_test_should_be_ok
	do_test_should_be_ng
	do_test_result_no_care
	if [ ! -e $BASE_DIR ]; then
		echo This is first time. >&2
		echo Saving this result as comparison base >&2
		mv $OUTPUT_DIR $BASE_DIR
	else
		show_diff
	fi
}

main
