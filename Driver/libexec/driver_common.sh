function xmp_print_help()
{
cat <<EOF
usage: $1 <OPTIONS> <INPUTFILE> ...

Compile Driver Options

   -o <file>         : place the output into <file>.
   -c                : compile and assemble, but do not link.
   -E                : preprocess only; do not compile, assemble or link.
   -v,--verbose      : print processing status.
   --version         : print version.
   -h,--help         : print usage.
   --show-env        : show environment variables.
   --tmp             : output parallel code (__omni_tmp__<file>).
   --dry             : only print processing status (not compile).
   --debug           : save intermediate files in __omni_tmp__.
   --stop-pp         : save intermediate files and stop after preprocess.
   --stop-frontend   : save intermediate files and stop after frontend.
   --stop-translator : save intermediate files and stop after translator.
   --stop-backend    : save intermediate files and stop after backend.
   --stop-compile    : save intermediate files and stop after compile.

Process Options

  --Wp[option] : Add preprocessor option.
  --Wf[option] : Add frontend option.
  --Wx[option] : Add Xcode translator option.
  --Wb[option] : Add backend option.
  --Wn[option] : Add native compiler option.
  --Wl[option] : Add linker option.

XcalableMP Options

  -omp,--openmp       : enable OpenMP.
  -xacc,--xcalableacc : enable XcalableACC.
  --scalasca-all      : output results in scalasca format for all directives.
  --scalasca          : output results in scalasca format for selected directives.
  --tlog-all          : output results in tlog format for all directives.
  --tlog              : output results in tlog format for selected directives.
EOF
}

function xmp_error_exit()
{
    echo "$0: error: $1"
    echo "compilation terminated."
    exit 1
}

function xmp_print_version()
{
    VERSION_FILE="${OMNI_HOME}/etc/version"
    if [ -f $VERSION_FILE ]; then
	cat $VERSION_FILE
	echo ""
    else
	xmp_error_exit "$VERSION_FILE not exist."
    fi
}

function xmp_show_env()
{
    CONF_FILE=${OMNI_HOME}/etc/xmpcc.conf
    if [ -f $CONF_FILE ]; then
	for val in `sed '/^[[:space:]]*$/d' ${CONF_FILE} | grep -v '^#' | awk -F= '{print $1}'`
	do
	    echo -n ${val}=\"
            eval echo -n \"\$$val\"
	    echo \"
	done
    else
	xmp_error_exit "$CONF_FILE not exist."
    fi
}

function xmp_set_parameters()
{
    local tmp_args=""

    for arg in "${@}"; do
	case $arg in
	    -o)
                OUTPUT_FLAG=true;;
            -c)
		ENABLE_LINKER=false;;
	    -E)
		ONLY_PP=true;;
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
	    --debug)
		ENABLE_DEBUG=true;;
            --stop-pp)
		STOP_PP=true
		VERBOSE=true;;
            --stop-frontend)
		STOP_FRONTEND=true
		VERBOSE=true;;
	    --stop-translator)
		STOP_TRANSLATOR=true
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
		XCODE_TRANSLATOR_ADD_OPT=${arg#--Wt}
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
		ENABLE_OPENMP=true;;
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
		    tmp_args="$tmp_args $arg"
		fi;;
	esac
    done

    if test $OUTPUT_TEMPORAL = true -a $DRY_RUN = true; then
        xmp_error_exit "cannot use both --tmp and --dry options at the same time."
    fi

    for arg in $tmp_args; do
	if [[ $arg =~ \.c$ ]]; then
            c_files="$c_files $arg"
	elif [[ $arg =~ \.a$ ]]; then
	    archive_files="$archive_files $arg"
	elif [[ "${arg}" =~ \.o$ ]]; then
            obj_files="$obj_files $arg"
	else
            other_args="$other_args $arg"
	fi
    done
}

function xmp_check_file_exist()
{
    ([ "$c_files" = "" ] && [ "$obj_files" = "" ]) && xmp_error_exit "no input files."
}

function xmp_exec()
{
    if [ $VERBOSE = true ] || [ $DRY_RUN = true ]; then
	echo $@
    fi

    if [ $DRY_RUN = false ]; then
	eval $@
    fi

    [ $? -ne 0 ] && { xmp_exec rm -rf $TEMP_DIR; exit 1; }
}

# ./hoge/fuga.c -> hoge_2f_fuga_2f_a
function xmp_norm_file_name()
{
    local NORM_NAME=`echo $1 | sed 's/^\.\///'`      # ./hoge/fuga.c -> hoge/fuga.c
    NORM_NAME=`echo $NORM_NAME | sed 's/\//_2f_/g'`  # hoge/fuga/a.c -> hoge_2f_fuga_2f_a.c        # "2f" is a hex number of '/'.
    NORM_NAME=`echo $NORM_NAME | sed 's/\./_2e_/g'`  # "." -> "_2e_"
    NORM_NAME=`basename $NORM_NAME .c`               # hoge_2f_fuga_2f_a.c -> hoge_2f_fuga_2f_a

    echo $NORM_NAME
}
