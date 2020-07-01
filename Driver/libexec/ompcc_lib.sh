function ompcc_print_help()
{
cat <<EOF
usage: $1 <OPTIONS> <INPUTFILE> ...

Compile Driver Options

   -o <file>         : place the output into <file>.
   -I <dir>          : add the directory dir to the list of directories to be searched for header files.
   -c                : compile and assemble, but do not link.
   -E                : preprocess only; do not compile, assemble or link.
   -v,--verbose      : print processing status.
   --version         : print version.
   -h,--help         : print usage.
   --show-env        : show environment variables.
   --tmp             : output translated code to __omni_tmp__<file>.
   --dry             : only print processing status (not compile).
   --debug           : save intermediate files to __omni_tmp__/.
   --cc=<CC>         : specify C compiler
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

Omni OpenACC Options

  -acc, --openacc         : enable OpenACC function.
  --no-ldg                : disable use of read-only data cache.
  --default-veclen=LENGTH : specify default vector length (default: 256)
  --platform=PLATFORM     : Specify platform [CUDA | OpenCL | PZCL] (default: $OPENACC_PLATFORM)
  --device=DEVICE         : Specify device [ccXX (XX is compute capability) | Fermi (=cc20) | Kepler (=cc35) | PEZYSC] (default: $OPENACC_DEVICE)

Omni OpenMP Options
  -fopenmp-only-target    : enable OpenMP only target function.
EOF
}

function ompcc_show_env()
{
    CONF_FILE=${OM_DRIVER_CONF_DIR}/ompcc.conf
    if [ -f "${CONF_FILE}" ]; then
	for val in `sed '/^[[:space:]]*$/d' "${CONF_FILE}" | grep -v '^#' | awk -F= '{print $1}'`
	do
	    echo -n ${val}=\"
            eval echo -n \"\$$val\"
	    echo \"
	done
    fi
}

function get_target()
{
    DIR=$(cd $(dirname $0); pwd)
    grep TARGET $DIR/../etc/xmpcc.conf | sed 's/TARGET=//' | sed "s/\"//g"
}

function ompcc_set_parameters()
{
    target=`get_target`
    while [ -n "$1" ]; do
        case "$1" in
            *.c)
                c_files+=("$1");;
            *.a)
                archive_files+=("$1");;
            *.o)
                obj_files+=("$1");;
            -o)
		shift;
		output_file=("$1");;
            -c)
		ENABLE_LINKER=false;;
	    -E)
		ONLY_PP=true;;
	    -D?*)
		define_opts+=("$1");;
	    -l?*)
		lib_args+=("$1");;
            -v|--verbose)
		VERBOSE=true;;
	    --version)
		omni_print_version; exit 0;;
            -h|--help)
		ompcc_print_help `basename $0`; exit 0;;
	    --show-env)
		ompcc_show_env;	exit 0;;
            --tmp)
		OUTPUT_TEMPORAL=true;;
            --dry)
		DRY_RUN=true;;
	    --debug)
		ENABLE_DEBUG=true;;
	    --cc=*)
		ENABLE_SPECIFY_CC=true
		SPECIFIED_CC="${1#--cc=}";;
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
	    -acc|--openacc)
		[ ${ENABLE_ACC} = "0" ] && omni_error_exit "warning: $1 option is unavailable, rebuild the compiler with ./configure --enable-openacc"
		ENABLE_ACC=true;;
	    -fopenmp-only-target)
		ENABLE_TARGET=true;;
	    --no-ldg)
		DISABLE_LDG=true;;
	    --default-veclen=*)
		DEFAULT_VECLEN="${1#--default-veclen=}";;
	    --platform=*)
		OPENACC_PLATFORM="${1#--platform=}";;
	    --device=*)
		OPENACC_DEVICE="${1#--device=}";;
            *)
		other_args+=("$1");;
	esac
	shift
    done

    if test $OUTPUT_TEMPORAL = true -a $DRY_RUN = true; then
        omni_error_exit "cannot use both --tmp and --dry options at the same time."
    fi
}
