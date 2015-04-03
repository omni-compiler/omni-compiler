#!/bin/bash

#-------------------------------------------------------
#  generator for xmp_coarray_alloc_wrap.f90
#-------------------------------------------------------

#--------------------
#  sub
#--------------------
echo72 () {
    str="$1                                                                        "
    str=`echo "$str" | cut -c -72`"&"
    echo "$str"
}


print_subr_alloc() {
    tk=$1
    typekind=$2

    echo72  "      subroutine xmpf_coarray_alloc${DIM}d_${tk}(descptr, var, count, element,"

 case "${DIM}" in
 0) echo    "     & tag, rank)" ;;
 1) echo    "     & tag, rank, lb1, ub1)" ;;
 2) echo    "     & tag, rank, lb1, ub1, lb2, ub2)" ;;
 3) echo    "     & tag, rank, lb1, ub1, lb2, ub2, lb3, ub3)" ;;
 4) echo    "     & tag, rank, lb1, ub1, lb2, ub2, lb3, ub3, lb4, ub4)" ;;
 5) echo    "     & tag, rank, lb1, ub1, lb2, ub2, lb3, ub3, lb4, ub4, lb5, ub5)" ;;
 6) echo72  "     & tag, rank, lb1, ub1, lb2, ub2, lb3, ub3, lb4, ub4, lb5, ub5,"
    echo    "     & lb6, ub6)" ;;
 7) echo72  "     & tag, rank, lb1, ub1, lb2, ub2, lb3, ub3, lb4, ub4, lb5, ub5,"
    echo    "     & lb6, ub6, lb7, ub7)" ;;
 esac
    echo    "        integer(8), intent(inout) :: descptr"
    echo    "        integer(8), intent(in) :: tag"
 case "${DIM}" in
 0) echo    "        integer, intent(in) :: count, element, rank" ;;
 1) echo    "        integer, intent(in) :: count, element, rank, lb1, ub1" ;;
 2) echo    "        integer, intent(in) :: count, element, rank, lb1, ub1, lb2, ub2" ;;
 3) echo72  "        integer, intent(in) :: count, element, rank, lb1, ub1, lb2, ub2,"
    echo    "     &    lb3, ub3" ;;
 4) echo72  "        integer, intent(in) :: count, element, rank, lb1, ub1, lb2, ub2,"
    echo    "     &    lb3, ub3, lb4, ub4" ;;
 5) echo72  "        integer, intent(in) :: count, element, rank, lb1, ub1, lb2, ub2,"
    echo    "     &    lb3, ub3, lb4, ub4, lb5, ub5" ;;
 6) echo72  "        integer, intent(in) :: count, element, rank, lb1, ub1, lb2, ub2,"
    echo    "     &    lb3, ub3, lb4, ub4, lb5, ub5, lb6, ub6" ;;
 7) echo72  "        integer, intent(in) :: count, element, rank, lb1, ub1, lb2, ub2,"
    echo    "     &    lb3, ub3, lb4, ub4, lb5, ub5, lb6, ub6, lb7, ub7" ;;
 esac
 case "${DIM}" in
 0) echo    "        ${typekind}, pointer, intent(out) :: var" ;;
 1) echo    "        ${typekind}, pointer, intent(out) :: var(:)" ;;
 2) echo    "        ${typekind}, pointer, intent(out) :: var(:,:)" ;;
 3) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:)" ;;
 4) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:)" ;;
 5) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:,:)" ;;
 6) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:,:,:)" ;;
 7) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:,:,:,:)" ;;
 esac
# START BODY OF PROCEDURE
 case "${DIM}" in
 0) echo    "        ${typekind} :: obj" ;;
 1) echo    "        ${typekind} :: obj(lb1:ub1) " ;;
 2) echo    "        ${typekind} :: obj(lb1:ub1, lb2:ub2) " ;;
 3) echo    "        ${typekind} :: obj(lb1:ub1, lb2:ub2, lb3:ub3) " ;;
 4) echo    "        ${typekind} :: obj(lb1:ub1, lb2:ub2, lb3:ub3, lb4:ub4) " ;;
 5) echo72  "        ${typekind} :: obj(lb1:ub1, lb2:ub2, lb3:ub3, lb4:ub4, "
    echo    "     &    lb5:ub5)" ;;
 6) echo72  "        ${typekind} :: obj(lb1:ub1, lb2:ub2, lb3:ub3, lb4:ub4, "
    echo    "     &    lb5:ub5, lb6:ub6)" ;;
 7) echo72  "        ${typekind} :: obj(lb1:ub1, lb2:ub2, lb3:ub3, lb4:ub4, "
    echo    "     &    lb5:ub5, lb6:ub6, lb7:ub7)" ;;
 esac
    echo    "        pointer (crayptr, obj)"
    echo    "        call xmpf_coarray_malloc(descptr, crayptr, count, element, tag)"
    echo    "        call pointer_assign(var, obj)"
    echo    "        return"
    echo    "      contains"
    echo    "        subroutine pointer_assign(p, d)"
 case "${DIM}" in
 0) echo    "          ${typekind}, pointer :: p"
    echo    "          ${typekind}, target  :: d" ;;
 1) echo    "          ${typekind}, pointer :: p(:)"
    echo    "          ${typekind}, target  :: d(:)" ;;
 2) echo    "          ${typekind}, pointer :: p(:,:)"
    echo    "          ${typekind}, target  :: d(:,:)" ;;
 3) echo    "          ${typekind}, pointer :: p(:,:,:)"
    echo    "          ${typekind}, target  :: d(:,:,:)" ;;
 4) echo    "          ${typekind}, pointer :: p(:,:,:,:)"
    echo    "          ${typekind}, target  :: d(:,:,:,:)" ;;
 5) echo    "          ${typekind}, pointer :: p(:,:,:,:,:)"
    echo    "          ${typekind}, target  :: d(:,:,:,:,:)" ;;
 6) echo    "          ${typekind}, pointer :: p(:,:,:,:,:,:)"
    echo    "          ${typekind}, target  :: d(:,:,:,:,:,:)" ;;
 7) echo    "          ${typekind}, pointer :: p(:,:,:,:,:,:,:)"
    echo    "          ${typekind}, target  :: d(:,:,:,:,:,:,:)" ;;
 esac
    echo    "          p => d"
    echo    "          return"
    echo    "        end subroutine"
# END BODY OF PROCEDURE
    echo    "      end subroutine"
    echo
}


print_subr_dealloc() {
    tk=$1
    typekind=$2

    echo    "      subroutine xmpf_coarray_dealloc${DIM}d_${tk}(descptr, var, tag)"
    echo    "        integer(8), intent(in) :: descptr, tag"

 case "${DIM}" in
 0) echo    "        ${typekind}, pointer, intent(out) :: var" ;;
 1) echo    "        ${typekind}, pointer, intent(out) :: var(:)" ;;
 2) echo    "        ${typekind}, pointer, intent(out) :: var(:,:)" ;;
 3) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:)" ;;
 4) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:)" ;;
 5) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:,:)" ;;
 6) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:,:,:)" ;;
 7) echo    "        ${typekind}, pointer, intent(out) :: var(:,:,:,:,:,:,:)" ;;
 esac

# START BODY OF PROCEDURE
    echo    "        nullify(var)"
    echo    "        call xmpf_coarray_free(descptr, tag)"
    echo    "        return"
# END BODY OF PROCEDURE

    echo    "      end subroutine"
    echo
}


#--------------------
#  main
#--------------------
echo "!! This file is automatically generated by xmp_coarray_alloc_wrap.f90.sh"
echo

for DIM in `seq 0 7`
do
    print_subr_alloc i2  "integer(2)"      
    print_subr_alloc i4  "integer(4)"      
    print_subr_alloc i8  "integer(8)"      
    print_subr_alloc l2  "logical(2)"      
    print_subr_alloc l4  "logical(4)"      
    print_subr_alloc l8  "logical(8)"      
    print_subr_alloc r4  "real(4)"         
    print_subr_alloc r8  "real(8)"         
    print_subr_alloc z8  "complex(4)"      
    print_subr_alloc z16 "complex(8)"      
    print_subr_alloc cn  "character(element)" 
done

for DIM in `seq 0 7`
do
    print_subr_dealloc i2  "integer(2)"      
    print_subr_dealloc i4  "integer(4)"      
    print_subr_dealloc i8  "integer(8)"      
    print_subr_dealloc l2  "logical(2)"      
    print_subr_dealloc l4  "logical(4)"      
    print_subr_dealloc l8  "logical(8)"      
    print_subr_dealloc r4  "real(4)"         
    print_subr_dealloc r8  "real(8)"         
    print_subr_dealloc z8  "complex(4)"      
    print_subr_dealloc z16 "complex(8)"      
    print_subr_dealloc cn  "character(*)" 
done

exit
