function xmpf90_print_help()
{
cat <<EOF
usage: $1 <OPTIONS> <INPUTFILE> ...

Compile Driver Options

   -o <file>         : place the output into <file>.
   -J <dir>          : specify where to put .mod and .xmod files for compiled modules.
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

function xmpf90_show_env()
{
    CONF_FILE=${OMNI_HOME}/etc/xmpf90.conf
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

function xmpf90_set_parameters()
{
    local tmp_args=
    local OUTPUT_FLAG=
    local MODULE_FLAG=

    for arg in "${@}"; do
	case $arg in
	    -o)
                OUTPUT_FLAG=true;;
	    -J)
		MODULE_FLAG=true;;
	    -J?*)
		MODULE_OPT="$MODULE_OPT -M${arg#-J}"
		other_args="$other_args $arg";;
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
		xmpf90_print_help $scriptname
		exit 0;;
	    --show-env)
		xmpf90_show_env
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
		elif [[ "$MODULE_FLAG" = true ]]; then
		    MODULE_OPT="$MODULE_OPT -M$arg"
		    other_args="$other_args -J $arg"
		    MODULE_FLAG=false
		else
		    tmp_args="$tmp_args $arg"
		fi;;
	esac
    done

    if test $OUTPUT_TEMPORAL = true -a $DRY_RUN = true; then
        xmp_error_exit "cannot use both --tmp and --dry options at the same time."
    fi

    for arg in $tmp_args; do
	if [[ $arg =~ \.F90$ ]] || [[ $arg =~ \.F$ ]]; then
	    F_F90_files="$F_F90_files $arg"
	    all_files="$all_files $arg"
	elif [[ $arg =~ \.f90$ ]] || [[ $arg =~ \.f$ ]]; then
            f_f90_files="$f_f90_files $arg"
	    all_files="$all_files $arg"
	elif [[ $arg =~ \.a$ ]]; then
	    archive_files="$archive_files $arg"
	elif [[ "${arg}" =~ \.o$ ]]; then
            obj_files="$obj_files $arg"
	else
            other_args="$other_args $arg"
	fi
    done
}

function xmpf90_check_file_exist()
{
    ([ "$all_files" == "" ] && [ "$obj_files" = "" ]) && xmp_error_exit "no input files."

    for file in $F_F90_files $f_f90_files $obj_files $archive_files; do
	if [ ! -f $file ]; then
	    xmp_error_exit "not found ${file}"
	fi
    done
}

# ./hoge/fuga.f90 -> hoge_2f_fuga
function xmpf90_norm_file_name()
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