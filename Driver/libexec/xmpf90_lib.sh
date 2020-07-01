function xmpf90_print_help()
{
cat <<EOF
usage: $1 <OPTIONS> <INPUTFILE> ...

Compile Driver Options

   -o <file>         : place the output into <file>.
   -I <dir>          : add the directory dir to the list of directories to be searched for header files.
   -J <dir>          : specify where to put .mod and .xmod files for compiled modules.
   -c                : compile and assemble, but do not link.
   -E                : preprocess only; do not compile, assemble or link.
   -cpp              : enable preprocess.
   -v,--verbose      : print processing status.
   --version         : print version.
   -h,--help         : print usage.
   --show-env        : show environment variables.
   --tmp             : output translated code to __omni_tmp__<file>.
   --dry             : only print processing status (not compile).
   --debug           : save intermediate files to __omni_tmp__/.
   --stop-pp         : save intermediate files and stop after preprocess.
   --stop-frontend   : save intermediate files and stop after frontend.
   --stop-translator : save intermediate files and stop after translator.
   --stop-backend    : save intermediate files and stop after backend.
   --stop-compile    : save intermediate files and stop after compile.

Process Options

  --Wp[option] : add preprocessor option.
  --Wf[option] : add frontend option.
  --Wx[option] : add Xcode translator option.
  --Wb[option] : add backend option.
  --Wn[option] : add native compiler option.
  --Wl[option] : add linker option.

XcalableMP Options

  -omp,--openmp                    : enable OpenMP function.
  -xacc[=pgi],--xcalableacc[=pgi]  : enable XcalableACC function. When adding "=pgi",
                                     PGI compiler is used as a backend OpenACC compiler.
  -max_assumed_shape=N             : specifies the maximum number of assumed-shape array arguments (default: 16)
EOF
}

function xmpf90_show_env()
{
    CONF_FILE=${OM_DRIVER_CONF_DIR}/xmpf90.conf
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

function get_target()
{
    DIR=$(cd $(dirname $0); pwd)
    grep TARGET $DIR/../etc/xmpf90.conf | sed 's/TARGET=//' | sed "s/\"//g"
}

function xmpf90_set_parameters()
{
    target=`get_target`
    while [ -n "$1" ]; do
        case "$1" in
	    *.f90|*.f|*.F90|*.F)
		f_files+=("$1");;
	    *.a)
		archive_files+=("$1");;
	    *.o)
		obj_files+=("$1");;
	    -o)
                shift;
		output_file=("$1");;
            -J)
		shift;
		module_dir=("${1#-J}")
		module_opt=("-M${module_dir[0]}")
		if [ "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu" -o "$target" = "FX100-linux-gnu" ]; then
		    module_dirs+=("${OMNI_MODINC}${module_dir}")
		elif [ "$target" = "sxace-nec-superux" ]; then
		    module_dir="${module_dir//\ /\\ }" # replace [space] -> \[space]
		    include_opt+=("-I" "${module_dir}")
		    module_dirs=("${OMNI_MODINC}" "${module_dir}")
		else
		    module_dirs+=("${OMNI_MODINC}" "${module_dir}")
		fi;;
	    -J?*)
		module_dir=("${1#-J}")
		module_opt=("-M${module_dir[0]}")
		if [ "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu" -o "$target" = "FX100-linux-gnu" ]; then
		    module_dirs+=("${OMNI_MODINC}${module_dir}")
		elif [ "$target" = "sxace-nec-superux" ]; then
		    tmodule_dir="${module_dir//\ /\\ }" # replace [space] -> \[space]
		    include_opt+=("-I" "${module_dir}")
		    module_dirs=("${OMNI_MODINC}" "${module_dir}")
		else
		    module_dirs+=("${OMNI_MODINC}" "${module_dir}")
		fi;;
            -I)
		shift; 
		include_opt+=("-I$1")
		other_args+=("-I$1")
		tmp_dir=("${1#-I}")
		trans_module_opt+=("-M${tmp_dir[0]}");;
	    -I?*)
		include_opt+=("$1")
		other_args+=("$1")
		tmp_dir=("${1#-I}")
		trans_module_opt+=("-M${tmp_dir[0]}");;
	    -l?*)
		lib_args+=("$1");;
	    -D?*)
		define_opts+=("$1");;
	    -ew) # Fix [xmp-bts:475]
		if [ "$target" = "sxace-nec-superux" ]; then
		    echo "On SX-ACE, \"-ew\" option cannot be used."
		    exit 1
		else
		    other_args+=("$1")
		fi;;
            -c)
		ENABLE_LINKER=false;;
	    -E)
		ONLY_PP=true;;
	    -cpp)
		ENABLE_CPP=true;;
            -v|--verbose)
		VERBOSE=true;;
	    --version)
		omni_print_version; exit 0;;
            -h|--help)
		xmpf90_print_help `basename $0`; exit 0;;
	    --show-env)
		xmpf90_show_env; exit 0;;
            --tmp)
		OUTPUT_TEMPORAL=true;;
            --dry)
		DRY_RUN=true;;
	    --debug)
		ENABLE_DEBUG=true;;
            --stop-pp)
		VERBOSE=true; STOP_PP=true;;
            --stop-frontend)
		VERBOSE=true; STOP_FRONTEND=true;;
	    --stop-translator)
		VERBOSE=true; STOP_TRANSLATOR=true;;
	    --stop-backend)
		VERBOSE=true; STOP_BACKEND=true;;
	    --stop-compile)
		VERBOSE=true; STOP_COMPILE=true;;
            --Wp*)
		local tmp=("${1#--Wp}")
		for v in $tmp
		do
		    pp_add_opt+=("$v")
		done;;
            --Wf*)
		local tmp=("${1#--Wf}")
		for v in $tmp
		do
		    frontend_add_opt+=("$v")
		done;;
            --Wx*)
		local tmp=("${1#--Wx}")
		for v in $tmp
		do
		    xcode_translator_add_opt+=("$v")
		done;;
            --Wn*)
		local tmp=("${1#--Wn}")
		for v in $tmp
		do
		    native_add_opt+=("$v")
		done;;
            --Wb*)
		local tmp=("${1#--Wb}")
		for v in $tmp
		do
		    backend_add_opt+=("$v")
		done;;
            --Wl*)
		local tmp=("${1#--Wl}")
		for v in $tmp
		do
		    linker_add_opt+=("$v")
		done;;
	    --openmp|-omp)
		ENABLE_OPENMP=true;;
	    -acc|--openacc)
		omni_error_exit "OpenACC for xmpf90 has been not implemented yet."
		if [ ${ENABLE_ACC} = "0" ]; then
		    omni_error_exit "warning: $arg option is unavailable, rebuild the compiler with ./configure --enable-openacc"
		fi
		ENABLE_OPENACC=true
		;;
            --xcalableacc|-xacc|--xcalableacc=*|-xacc=*)
		local tmp=${1}
		tmp=${tmp#*=}
		if [ "pgi" = "$tmp" ]; then
		    other_args+=("-acc")
		elif [ "-xacc" = "$tmp" ]; then
		    : #do nothing
		else
		    omni_error_exit "Unknown OpenACC compiler for XACC"
		fi
		ENABLE_XACC=true;;
	    --noopenacc|-noacc)
		ENABLE_OPENACC=false;;
	    --scalasca-all)
		ENABLE_SCALASCA_ALL=true;;
	    --scalasca)
		ENABLE_SCALASCA=true;;
	    --tlog-all)
		ENABLE_TLOG_ALL=true;;
	    --tlog)
		ENABLE_TLOG=true;;
	    -Mnolarge_arrays)
		omni_error_exit "-Mnolarge_arrays option cannot be used.";;
	    *)
		other_args+=("$1");;
	esac
	shift
    done

    if test $OUTPUT_TEMPORAL = true -a $DRY_RUN = true; then
        omni_error_exit "cannot use both --tmp and --dry options at the same time."
    fi
}
