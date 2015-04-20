SUBROUTINE checkp ( ire , ians , error )
# 3 "fortranscope.f90"
 INTEGER :: ire
# 3 "fortranscope.f90"
 INTEGER :: error
# 3 "fortranscope.f90"
 INTEGER :: ians
 INTEGER :: xmp_size_array ( 0 : 15 , 0 : 6 )
 COMMON / XMP_COMMON / xmp_size_array

 CALL xmpf_checkp ( ire , ians , error )
99999 &
 CONTINUE

CONTAINS
 SUBROUTINE xmpf_checkp ( ire , ians , error )
  INTEGER :: ire
  INTEGER :: error
  INTEGER :: ians

# 5 "fortranscope.f90"
  IF ( ire /= ians ) THEN
# 6 "fortranscope.f90"
   error = error + 1
  END IF
  GOTO 99999
 99999 &
  CONTINUE
 END SUBROUTINE xmpf_checkp

END SUBROUTINE checkp

