function omni_set_trap()
{
  trap "rm -rf $TEMP_DIR; exit 1" 2
}

function omni_error_exit()
{
    echo "$0: error: $1"
    echo "compilation terminated."
    exit 1
}

function omni_print_version()
{
    VERSION_FILE="${OM_DRIVER_CONF_DIR}/version"
    if [ -f $VERSION_FILE ]; then
	cat $VERSION_FILE
	echo ""
    else
	omni_error_exit "$VERSION_FILE not exist."
    fi
}

function omni_exec_echo()
{
    [ $VERBOSE = true -o $DRY_RUN = true ] && echo ${@+"$@"}
}

function omni_exec_run()
{
    [ $DRY_RUN = false ] && ${@+"$@"}
}

function error_process()
{
    [ $ENABLE_DEBUG = false ] && omni_exec rm -rf $TEMP_DIR
    exit 1
}

function omni_exec()
{
    [ $VERBOSE = true -o $DRY_RUN = true ] && echo ${@+"$@"}
    if [ $DRY_RUN = false ]; then
	${@+"$@"} || error_process
    fi
}

function omni_f_check_file_exist()
{
    ( [ "${f_files}" = "" ] && [ "${obj_files}" = "" ] ) && omni_error_exit "no input files."

    for file in "${f_files[@]}" "${obj_files[@]}" "${archive_files[@]}"; do
	[ ! -f "${file}" ] && omni_error_exit "not found ${file}"
    done
}

# ./hoge/fuga-a.f90 -> hoge_2f_fuga_2d_a
# http://d.hatena.ne.jp/eggtoothcroc/20110718/p1
function omni_f_norm_file_name()
{
    local filename=`basename "$1" .f90`          # ./hoge/fuga-a.f90 -> fuga-a
    filename=`basename "$filename" .F90`
    filename=`basename "$filename" .f`
    filename=`basename "$filename" .F`
    filename=`dirname "$1"`/${filename}          #  -> ./hoge/fuga-a
    filename=`echo $filename | sed 's/^\.\///'`  #         -> hoge/fuga-a

    local len=${#filename}
    local NORM_NAME=""
    for i in `seq 1 $len`; do             #  -> hoge_2f_fuga_2d_a
        c=`echo $filename | cut -c $i`
        ix=$(printf "%d" \'$c)
        if [ $ix -ge 48 -a $ix -le 57 ]; then    # '0' <= $ix <= '9'
            NORM_NAME="$NORM_NAME"$c
        elif [ $ix -ge 65 -a $ix -le 90 ]; then  # 'A' <= $ix <= 'Z'
            NORM_NAME="$NORM_NAME"$c
        elif [ $ix -ge 97 -a $ix -le 122 ]; then # 'a' <= $ix <= 'z'
            NORM_NAME="$NORM_NAME"$c
        else
            local hex=$(printf "%x" $ix )
            local char=$(printf "\x${hex}")
            local code=\'$char
            NORM_NAME="$NORM_NAME"`printf "_%x_\n" $code`
        fi
    done

    echo "${NORM_NAME}"
}


function omni_c_check_file_exist()
{
    ([ "${c_files}" = "" ] && [ "${obj_files}" = "" ]) && omni_error_exit "no input files."

    for file in "${c_files[@]}" "${obj_files[@]}"; do
        [ ! -f "${file}" ] && omni_error_exit "not found ${file}"
    done
}

# ./hoge/fuga-a.c -> hoge_2f_fuga_2d_a
# http://d.hatena.ne.jp/eggtoothcroc/20110718/p1
function omni_c_norm_file_name()
{
    local filename=`basename "$1" .c`    # ./hoge/fuga-a.c -> fuga-a
    filename=`dirname "$1"`/${filename}  #                 -> ./hoge/fuga-a
    filename=`echo $filename | sed 's/^\.\///'`  #         -> hoge/fuga-a

    local len=${#filename}
    local NORM_NAME=""
    for i in `seq 1 $len`; do                    #         -> hoge_2f_fuga_2d_a
	c=`echo $filename | cut -c $i`
	ix=$(printf "%d" \'$c)
	if [ $ix -ge 48 -a $ix -le 57 ]; then    # '0' <= $ix <= '9'
	    NORM_NAME="$NORM_NAME"$c
	elif [ $ix -ge 65 -a $ix -le 90 ]; then  # 'A' <= $ix <= 'Z'
	    NORM_NAME="$NORM_NAME"$c
	elif [ $ix -ge 97 -a $ix -le 122 ]; then # 'a' <= $ix <= 'z'
	    NORM_NAME="$NORM_NAME"$c
	else
	    local hex=$(printf "%x" $ix )
	    local char=$(printf "\x${hex}")
	    local code=\'$char
	    NORM_NAME="$NORM_NAME"`printf "_%x_\n" $code`
	fi
    done

    echo "${NORM_NAME}"
}
