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
