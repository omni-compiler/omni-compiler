function xmp_print_help()
{
cat <<EOF
usage: $1 <OPTIONS> <INPUTFILE> ...

Compile Driver Options

   -o <file>       : place the output into <file>
   -c              : compile and assemble, but do not link
   -E              : preprocess only; do not compile, assemble or link (as the same --stop-pp)
   -v,--verbose    : print processing status.
   --version       : print version.
   -h,--help       : print usage.
   --show-env      : show environment variables.
   --tmp           : output parallel code (__omni_tmp_SRCNAME.c).
   --dry           : only print processing status (not compile).
   --stop-pp       : save intermediate file and stop after preprocess.
   --stop-frontend : save intermediate file and stop after frontend.
   --stop-xcode    : save intermediate file and stop after Xcode process.
   --stop-backend  : save intermediate file and stop after backend.
   --stop-compile  : save intermediate file and stop after native compile.

Process Options

  --Wp[option] : Add preprocessor option.
  --Wf[option] : Add frontend option.
  --Wx[option] : Add Xcode processor option.
  --Wb[option] : Add backend option.
  --Wn[option] : Add native compiler option.
  --Wl[option] : Add linker option.

XcalableMP Options

  -omp,--openmp       : enable OpenMP function.
  -xacc,--xcalableacc : enable XcalableACC.
  --scalasca-all      : output results in scalasca format of all directives.
  --scalasca          : output results in scalasca format of selected directives.
  --tlog-all          : output results in tlog format of all directives.
  --tlog              : output results in tlog format of selected directives.
EOF
}

function xmp_print_version()
{
    echo "version"
}

function xmp_error_exit()
{
    echo "$0: error: $1"
    echo "compilation terminated."
    exit 1
}

function xmp_show_env()
{
    for val in `sed '/^[[:space:]]*$/d' ${OMNI_HOME}/etc/xmpcc.conf | grep -v '^#' | awk -F= '{print $1}'`
    do
	echo -n ${val}=\"
        eval echo -n \"\$$val\"
	echo \"
    done
}

function xmp_set_parameters()
{
    for arg in "${@}"; do
	case $arg in
	    -o)
                OUTPUT_FLAG=true;;
            -c)
		ENABLE_LINKER=false;;
            -v|--verbose)
		VERBOSE=true;;
	    --version)
		xmp_print_version
		exit 0;;
            -h|--help)
		local scriptname=`basename $0`
		xmp_print_help $scriptname
		exit 0;;
	    --show-env)
		xmp_show_env
		exit 0;;
            --tmp)
		OUTPUT_TEMPORAL=true;;
            --dry)
		DRY_RUN=true;;
            --stop-pp|-E)
		STOP_PP=true
		VERBOSE=true;;
            --stop-frontend)
		STOP_FRONTEND=true
		VERBOSE=true;;
	    --stop-xcode)
		STOP_XCODE=true
		VERBOSE=true;;
	    --stop-backend)
		STOP_BACKEND=true
		VERBOSE=true;;
	    --stop-compile)
		STOP_COMPILE=true
		VERBOSE=true;;
	    --Wp*)
		PP_ADD_OPT=${arg#--Wp}
                ;;
            --Wf*)
		FRONTEND_ADD_OPT=${arg#--Wf}
                ;;
            --Wx*)
		XCODE_ADD_OPT=${arg#--Wx}
                ;;
	    --Wn*)
		NATIVE_ADD_OPT=${arg#--Wn}
		;;
            --Wb*)
		BACKEND_ADD_OPT=${arg#--Wb}
                ;;
            --Wl*)
		LINKER_ADD_OPT=${arg#--Wl}
		;;
	    --openmp|-omp)
		compile_args="$compile_args $OPENMP_OPT";;
	    --xcalableacc|-xacc)
		ENABLE_XACC=true;;
	    --scalasca-all)
		ENABLE_SCALASCA_ALL=true;;
	    --scalasca)
		ENABLE_SCALASCA=true;;
	    --tlog-all)
		ENABLE_TLOG_ALL=true;;
	    --tlog)
		ENABLE_TLOG=true;;
            *)
		if [ "$OUTPUT_FLAG" = true ]; then
		    OUTPUT_FILE=$arg
		    OUTPUT_FLAG=false
		else
		    compile_args="$compile_args $arg"
		fi;;
	esac
    done

    if test $OUTPUT_TEMPORAL = true -a $DRY_RUN = true; then
        xmp_error_exit "cannot use both --tmp and --dry options at the same time."
    fi

    for arg in $compile_args; do
	if [[ $arg =~ \.c$ ]]; then
            c_files="$c_files $arg"
	elif [[ "${arg}" =~ \.o$ ]]; then
            obj_files="$obj_files $arg"
	else
            other_args="$other_args $arg"
	fi
    done
}

function xmp_check_file_exist()
{
    [ "$c_files" = "" ] && xmp_error_exit "no input files."
}

function xmp_exec()
{
    if [ $VERBOSE = true ] || [ $DRY_RUN = true ]; then
	echo $@
    fi

    if [ $DRY_RUN = false ]; then
	eval $@
    fi

    if test $? -ne 0; then
	exit 1
    fi
}

# /hoge/fuga/a.c -> hoge-fuga-a
function xmp_get_file_prefix()
{
    local NORM_NAME=`echo $1 | sed 's/^\.\///'`      # ./hoge/fuga.c -> hoge/fuga.c
    NORM_NAME=`echo $NORM_NAME | sed 's/\//_2f_/g'`  # hoge/fuga/a.c -> hoge_2f_fuga_2f_a.c # "2f" is a hex number of '/'
    NORM_NAME=`basename $NORM_NAME .c`               # hoge_2f_fuga_2f_a.c -> hoge_2f_fuga_2f_a

    echo $NORM_NAME
}
