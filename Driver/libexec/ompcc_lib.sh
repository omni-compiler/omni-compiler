function ompcc_print_help()
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

Omni OpenACC Options

  -acc, --openacc : Enable OpenACC.
EOF
}

function ompcc_show_env()
{
    CONF_FILE=${OMNI_HOME}/etc/ompcc.conf
    if [ -f $CONF_FILE ]; then
	for val in `sed '/^[[:space:]]*$/d' ${CONF_FILE} | grep -v '^#' | awk -F= '{print $1}'`
	do
	    echo -n ${val}=\"
            eval echo -n \"\$$val\"
	    echo \"
	done
    else
	omni_error_exit "$CONF_FILE not exist."
    fi
}

function ompcc_set_parameters()
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
		ompcc_print_version
		exit 0;;
            -h|--help)
		local scriptname=`basename $0`
		ompcc_print_help $scriptname
		exit 0;;
	    --show-env)
		ompcc_show_env
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
		XCODE_TRANSLATOR_ADD_OPT=${arg#--Wx}
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
	    -acc|--openacc)
		if [ ${ENABLE_ACC} = "0" ]; then
		    omni_error_exit "warning: $arg option is unavailable, rebuild the compiler with ./configure --enable-openacc"
		fi
		ENABLE_ACC=true
		;;
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
        omni_error_exit "cannot use both --tmp and --dry options at the same time."
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

