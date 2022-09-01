function omni_acc_cuda_get_args() # args, ...
{
    local args=($@)
    local rets=()

    for arg in ${args[@]}; do
        case ${arg} in
            -g | -O[0-9])
                rets+=(${arg});;
            -fPIC | -fpic)
                rets+=(-Xcompiler ${arg});;
        esac
    done

    echo ${rets[@]}
}

function omni_acc_cuda_compile() # FILE_CU, FILE_O, args, ...
{
    local FILE_CU=$1
    shift
    local FILE_O=$1
    shift
    local ARGS=($@)

    local FILE_O_GPU="${FILE_O}".gpu
    local FILE_O_CPU="${FILE_O}".cpu

    omni_exec $OMNI_GPUCC_CMD $OMNI_GPUCC_OPT ${ARGS[@]} "${FILE_CU}" -o "${FILE_O_GPU}"
    omni_exec mv "${FILE_O}" "${FILE_O_CPU}"
    omni_exec ld -r "${FILE_O_CPU}" "${FILE_O_GPU}" -o "${FILE_O}"
    omni_exec rm -f "${FILE_O_CPU}"
}

function omni_acc_cuda_get_cc_version() # device_name
{
    case "${1}" in
        Fermi)
            echo 20;;
        Kepler)
            echo 35;;
        Maxwell)
            echo 52;;
        Pascal)
            echo 60;;
        Volta)
            echo 70;;
        cc*)
            echo "${1#cc}";;
        CC*)
            echo "${1#CC}";;
        *)
            return 1
    esac
}

function omni_acc_opencl_compile() # FILE_CU, FILE_O, args, ...
{
    local FILE_CL=$1
    shift
    local FILE_O=$1
    shift
    local ARGS=($@)

    local FILE_O_GPU="${FILE_O}".gpu
    local FILE_O_CPU="${FILE_O}".cpu
    local FILE_CL_C="${FILE_CL}".c
    local FILE_1=$(mktemp)
    local FILE_2=$(mktemp)

    cat "${OMNI_HOME}/include/acc_cl_hdr.cl" "${OMNI_HOME}/include/acc_cl_reduction.cl"  "${FILE_CL}" >  "${FILE_1}"
    omni_exec cpp "${FILE_1}" -o  "${FILE_2}"
    omni_exec $OMNI_OPENCL_CC_CMD $OMNI_OPENCL_CC_OPT ${ARGS[@]} -o "${FILE_CL_C}" "${FILE_2}"
    omni_exec $OMNI_CC_CMD -c -o "${FILE_O_GPU}" "${FILE_CL_C}"
    omni_exec mv "${FILE_O}" "${FILE_O_CPU}"
    omni_exec ld -r "${FILE_O_CPU}" "${FILE_O_GPU}" -o "${FILE_O}"
    omni_exec rm -f "${FILE_O_CPU}"
    omni_exec rm -f "${FILE_1}" "${FILE_2}" 
}
