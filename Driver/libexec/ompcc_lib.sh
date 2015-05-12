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
  --no-ldg        : Disable use of read-only data cache.
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

function ompcc_set_parameters()
{
    while [ -n "$1" ]; do
        case "$1" in
            *.c)
                c_files+=("$1");;
            *.a)
                archive_files+=("$1");;
            *.o)
                obj_files+=("$1");;
            -o)
                shift; output_file=("$1");;
            -c)
		ENABLE_LINKER=false;;
	    -E)
		ONLY_PP=true;;
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
                pp_add_opt+=("${1#--Wp}");;
            --Wf*)
                frontend_add_opt+=("${1#--Wf}");;
            --Wx*)
                xcode_translator_add_opt+=("${1#--Wx}");;
            --Wn*)
                native_add_opt+=("${1#--Wn}");;
            --Wb*)
                backend_add_opt+=("${1#--Wb}");;
            --Wl*)
                linker_add_opt+=("${1#--Wl}");;
	    -acc|--openacc)
		[ ${ENABLE_ACC} = "0" ] && omni_error_exit "warning: $1 option is unavailable, rebuild the compiler with ./configure --enable-openacc"
		ENABLE_ACC=true;;
	    --no-ldg)
		DISABLE_LDG=true;;
            *)
		other_args+=("$1");;
	esac
	shift
    done

    if test $OUTPUT_TEMPORAL = true -a $DRY_RUN = true; then
        omni_error_exit "cannot use both --tmp and --dry options at the same time."
    fi
}
