function omni_error_exit()
{
    echo "$0: error: $1"
    echo "compilation terminated."
    exit 1
}

function omni_print_version()
{
    VERSION_FILE="${OMNI_HOME}/etc/version"
    if [ -f $VERSION_FILE ]; then
	cat $VERSION_FILE
	echo ""
    else
	omni_error_exit "$VERSION_FILE not exist."
    fi
}

function omni_exec()
{
    if [ $VERBOSE = true ] || [ $DRY_RUN = true ]; then
	echo $@
    fi

    if [ $DRY_RUN = false ]; then
	eval $@
    fi

    [ $? -ne 0 ] && { omni_exec rm -rf $TEMP_DIR; exit 1; }
}


function omni_f_check_file_exist()
{
    ([ "$all_files" = "" ] && [ "$obj_files" = "" ]) && omni_error_exit "no input files."

    for file in $all_files $obj_files $archive_files; do
	if [ ! -f $file ]; then
	        omni_error_exit "not found ${file}"
fi
    done
}

# ./hoge/fuga.f90 -> hoge_2f_fuga
function omni_f_norm_file_name()
{
    local NORM_NAME=`basename $1 .f90`       # ./hoge/fuga.f90 -> ./hoge/fuga
    NORM_NAME=`basename $NORM_NAME .F90`
    NORM_NAME=`basename $NORM_NAME .f`
    NORM_NAME=`basename $NORM_NAME .F`
    local DIR=`dirname $1`
    NORM_NAME=${DIR}/${NORM_NAME}

    NORM_NAME=`echo $NORM_NAME | sed 's/^\.\///'`    # ./hoge/fuga -> hoge/fuga
    NORM_NAME=`echo $NORM_NAME | sed 's/\//_2f_/g'`  # hoge/fuga -> hoge_2f_fuga
                                                     # "2f" is a hex number of '/'.
    NORM_NAME=`echo $NORM_NAME | sed 's/\./_2e_/g'`  # "." -> "_2e_"

    echo $NORM_NAME
}


function omni_c_check_file_exist()
{
    ([ "$c_files" = "" ] && [ "$obj_files" = "" ]) && omni_error_exit "no input files."

    for file in $c_files $obj_files; do
        if [ ! -f $file ]; then
            omni_error_exit "not found ${file}"
        fi
    done
}

# ./hoge/fuga.c -> hoge_2f_fuga
function omni_c_norm_file_name()
{
    local NORM_NAME=`basename $1 .c`   # ./hoge/fuga.c -> ./hoge/fuga
    local DIR=`dirname $1`
    NORM_NAME=${DIR}/${NORM_NAME}
    NORM_NAME=`echo $NORM_NAME | sed 's/^\.\///'`    # ./hoge/fuga -> hoge/fuga
    NORM_NAME=`echo $NORM_NAME | sed 's/\//_2f_/g'`  # hoge/fuga -> hoge_2f_fuga
                                                     # "2f" is a hex number of '/'.
    NORM_NAME=`echo $NORM_NAME | sed 's/\./_2e_/g'`  # "." -> "_2e_"

    echo $NORM_NAME
}

