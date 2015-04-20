SUBROUTINE c ( i , j )
# 3 "bug1OK.f90"
 INTEGER :: j
# 3 "bug1OK.f90"
 INTEGER :: i
 INTEGER :: xmp_size_array ( 0 : 15 , 0 : 6 )
 COMMON / XMP_COMMON / xmp_size_array

 CALL xmpf_c ( i , j )
99999 &
 CONTINUE

CONTAINS
 SUBROUTINE xmpf_c ( i , j )
  INTEGER :: j
  INTEGER :: i

# 5 "bug1OK.f90"
  IF ( i /= j ) THEN
# 6 "bug1OK.f90"
   CONTINUE
  END IF
# 8 "bug1OK.f90"
  CONTINUE
 99999 &
  CONTINUE
 END SUBROUTINE xmpf_c

END SUBROUTINE c

