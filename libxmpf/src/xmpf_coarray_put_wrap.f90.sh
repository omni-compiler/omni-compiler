#!/bin/bash

#-------------------------------------------------------
#  generator for xmp_coarray_put_wrap.f90, 
#  PUT INTERFACE TYPE 8
#  see also ../include/xmp_coarray_put.h{,.sh}
#-------------------------------------------------------

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
    echo   "      subroutine xmpf_coarray_put0d_${tk}(descptr, coindex, mold, src)"
    echo   '!-----------------------------------------------------------------------'
    echo   '      integer(8), intent(in) :: descptr'
    echo   '      integer, intent(in) :: coindex'
    echo   "      ${typekind}, intent(in) :: mold, src"
    echo   ''

# ERROR CHECK
    if [ "${typekind}" == "character(*)" ]; then
        echo   '      if (len(mold).ne.len(src)) then'
        echo72 '        call xmpf_coarray_put_err_len(descptr,'
        echo   '     &    len(mold), len(src))'
        echo   '      end if'
        echo   ''
    fi

# EXECUTION
    echo72 "      call xmpf_coarray_put_scalar(descptr, loc(mold), ${element},"
    echo   '     &   coindex, src)'
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
    echo    "      subroutine xmpf_coarray_put${DIM}d_${tk}(descptr, coindex, mold, src)"
    echo    '!-----------------------------------------------------------------------'
    echo    '      integer(8), intent(in) :: descptr'
    echo    '      integer, intent(in) :: coindex'

    echo -n "      ${typekind}, intent(in) :: mold("
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n "${sep}:"
        sep=','
    done
    echo    ')'

    echo -n "      ${typekind}, intent(in) :: src("
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n "${sep}:"
        sep=','
    done
    echo    ')'

    echo    '      integer(8) :: base, base_s'
    echo    "      integer :: i, skip(${DIM}), skip_s(${DIM}), extent(${DIM})"
    echo    ''

# ERROR CHECK
    if [ "${typekind}" == "character(*)" ]; then
        echo    '      if (len(mold).ne.len(src)) then'
        echo72  '        call xmpf_coarray_put_err_len(descptr,'
        echo    '     &    len(mold), len(src))'
        echo    '      end if'
    fi
    echo    "      do i = 1, ${DIM}"
    echo    '        if (size(mold,i).ne.size(src,i)) then'
    echo72  '          call xmpf_coarray_put_err_size(descptr,'
    echo    '     &      i, size(mold,i), size(src,i))'
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

    echo -n '      base_s = loc(src('
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n ${sep}1
        sep=','
    done
    echo    '))'

    for i in `seq 1 ${DIM}`; do
        echo -n "      skip_s($i) = int( loc(src("
        sep=''
        for j in `seq 1 ${DIM}`; do
            if test $i -eq $j; then
                echo -n ${sep}2
            else
                echo -n ${sep}1
            fi
            sep=','
        done
        echo ")) - base_s )"
    done

    echo   ''
    echo72 "      call xmpf_coarray_put_array(descptr, base, ${element},"
    echo   "     &   coindex, src, ${DIM}, skip, skip_s, extent)"
    echo   '      return'
    echo   '      end subroutine'
    echo   ''
}


print_subroutine_spread() {
    tk="$1"
    typekind="$2"
    element="$3"

# DECLARATION
    echo    '!-----------------------------------------------------------------------'
    echo    "      subroutine xmpf_coarray_spread${DIM}d_${tk}(descptr, coindex, mold, src)"
    echo    '!-----------------------------------------------------------------------'
    echo    '      integer(8), intent(in) :: descptr'
    echo    '      integer, intent(in) :: coindex'

    echo -n "      ${typekind}, intent(in) :: mold("
    sep=''
    for i in `seq 1 ${DIM}`; do
        echo -n "${sep}:"
        sep=','
    done
    echo    ')'

    echo    "      ${typekind}, intent(in) :: src"

    echo    '      integer(8) :: base, base_s'
    echo    "      integer :: i, skip(${DIM}), extent(${DIM})"
    echo    ''

# ERROR CHECK
    if [ "${typekind}" == "character(*)" ]; then
        echo    '      if (len(mold).ne.len(src)) then'
        echo72  '        call xmpf_coarray_put_err_len(descptr,'
        echo    '     &    len(mold), len(src))'
        echo    '      end if'
        echo    ''
    fi

# EXECUTION
    echo    "      do i = 1, ${DIM}"
    echo    '        extent(i) = size(mold,i)'
    echo    '      end do'

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

    echo   ''
    echo72 "      call xmpf_coarray_put_spread(descptr, base, ${element},"
    echo   "     &   coindex, src, ${DIM}, skip, extent)"
    echo   '      return'
    echo   '      end subroutine'
    echo   ''
}


print_subroutine() {
    case ${DIM} in
        0) print_subroutine_scalar "$@" ;;
        *) print_subroutine_array  "$@"
           print_subroutine_spread "$@" ;;
    esac
}

#--------------------
#  main
#--------------------

TARGET=$1

echo "!! This file is automatically generated by $0"
echo '!! PUT INTERFACE TYPE 8'
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
