#!/bin/bash

#-------------------------------------------------------
#  generator for xmp_coarray_getsub_wrap.f90
#  *** OPTIMIZATION ROUTINE ***
#-------------------------------------------------------

#DEBUG=1

#--------------------
#  sub
#--------------------
echo72 () {
    str="$1                                                                        "
    str=`echo "$str" | cut -c -72`"&"
    echo "$str"
}

print_subroutine_scalar() {
    tk="$1"
    typekind="$2"
    element="$3"

# DECLARATION
    echo   '!-----------------------------------------------------------------------'
    echo   "      subroutine xmpf_coarray_getsub0d_${tk}(descptr, coindex, mold, dst)"
    echo   '!-----------------------------------------------------------------------'
    echo   '      integer(8), intent(in) :: descptr'
    echo   '      integer, intent(in) :: coindex'
#################################3
###    echo   "      ${typekind}, intent(inout) :: mold   !! 'inout' to avoid excessive code motion"
#################################3
    echo   "      ${typekind}, intent(in) :: mold"
    echo   "      ${typekind}, intent(out) :: dst"
    echo   ''

# DEBUG MESSAGE
    if [ "${DEBUG}" == "1" ]; then
        echo   "      print *, \"SELECTED SPECIFIC SUBROUTINE:\""
        echo   "      print *, \" xmpf_coarray_getsub0d_${tk}(descptr, coindex, mold, dst)\""
    fi

# ERROR CHECK
    if [ "${typekind}" == "character(*)" ]; then
        echo   '      if (len(mold).ne.len(dst)) then'
        echo72 '        call xmpf_coarray_getsub_err_len(descptr,'
        echo   '     &    len(mold), len(dst))'
        echo   '      end if'
        echo   ''
    fi

# EXECUTION
    echo72 "      call xmpf_coarray_get_scalar(descptr, loc(mold), ${element},"
    echo   '     &   coindex, dst)'
    echo   '      return'
    echo   '      end subroutine'
    echo   ''
}


print_subroutine_array() {
    tk="$1"
    typekind="$2"
    element="$3"

# DECLARATION
    echo    '!-----------------------------------------------------------------------'
    echo    "      subroutine xmpf_coarray_getsub${DIM}d_${tk}(descptr, coindex, mold, dst)"
    echo    '!-----------------------------------------------------------------------'
    echo    '      integer(8), intent(in) :: descptr'
    echo    '      integer, intent(in) :: coindex'

    echo -n "      ${typekind}, intent(in) :: mold("
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n "${sep}:"
        sep=','
    done
    echo ')'

    echo -n "      ${typekind}, intent(out) :: dst("
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n "${sep}:"
        sep=','
    done
    echo    ')'

    echo    '      integer(8) :: base, base_d'
    echo    "      integer :: i, skip(${DIM}), skip_d(${DIM}), extent(${DIM})"
    echo    ''

# DEBUG MESSAGE
    if [ "${DEBUG}" == "1" ]; then
        echo   "      print *, \"SELECTED SPECIFIC SUBROUTINE:\""
        echo   "      print *, \" xmpf_coarray_getsub${DIM}d_${tk}(descptr, coindex, mold, dst)\""
    fi

# ERROR CHECK
    if [ "${typekind}" == "character(*)" ]; then
        echo    '      if (len(mold).ne.len(dst)) then'
        echo72  '        call xmpf_coarray_getsub_err_len(descptr,'
        echo    '     &    len(mold), len(dst))'
        echo    '      end if'
    fi
    echo    "      do i = 1, ${DIM}"
    echo    '        if (size(mold,i).ne.size(dst,i)) then'
    echo72  '          call xmpf_coarray_getsub_err_size(descptr,'
    echo    '     &      i, size(mold,i), size(dst,i))'
    echo    '        end if'
    echo    '        extent(i) = size(mold,i)'
    echo    '      end do'
    echo    ''

# EXECUTION
    echo -n '      base = loc(mold('
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n ${sep}1
        sep=','
    done
    echo    '))'

    for i in `seq 1 ${DIM}`; do
        echo -n "      skip($i) = int( loc(mold("
        sep=''
        for j in `seq 1 ${DIM}`; do
            if test $i -eq $j; then
                echo -n ${sep}2
            else
                echo -n ${sep}1
            fi
            sep=','
        done
        echo ")) - base )"
    done

    echo -n '      base_d = loc(dst('
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n ${sep}1
        sep=','
    done
    echo    '))'

    for i in `seq 1 ${DIM}`; do
        echo -n "      skip_d($i) = int( loc(dst("
        sep=''
        for j in `seq 1 ${DIM}`; do
            if test $i -eq $j; then
                echo -n ${sep}2
            else
                echo -n ${sep}1
            fi
            sep=','
        done
        echo ")) - base_d )"
    done

    echo   ''
    echo72 "      call xmpf_coarray_getsub_array(descptr, base, ${element},"
    echo   "     &   coindex, base_d, ${DIM}, skip, skip_d, extent)"
    echo   '      return'
    echo   '      end subroutine'
    echo   ''
}


print_subroutine() {
    case ${DIM} in
        0) print_subroutine_scalar "$@" ;;
        *) print_subroutine_array  "$@" ;;
    esac
}


#--------------------
#  main
#--------------------
TARGET=$1

echo "!! This file is automatically generated by $0"
echo '!! GETSUB INTERFACE (OPTIMIZATION)'
echo

for DIM in `seq 0 7`
do
    if test "sxace-nec-superux" != "$TARGET"; then    ## integer(2) cannot be used on SX-ACE
	print_subroutine i2  "integer(2)"     2
    fi
    print_subroutine i4  "integer(4)"     4
    print_subroutine i8  "integer(8)"     8
    if test "sxace-nec-superux" != "$TARGET"; then    ## logical(2) cannot be used on SX-ACE
	print_subroutine l2  "logical(2)"     2
    fi
    print_subroutine l4  "logical(4)"     4
    print_subroutine l8  "logical(8)"     8
    print_subroutine r4  "real(4)"        4
    print_subroutine r8  "real(8)"        8
    print_subroutine z8  "complex(4)"     8
    print_subroutine z16 "complex(8)"     16
    print_subroutine cn  "character(*)"   "len(mold)"
done

exit
