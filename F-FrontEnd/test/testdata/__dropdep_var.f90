      MODULE dropdep_var
        IMPLICIT NONE
        PRIVATE
        PUBLIC :: r, chr1
        INTEGER,PARAMETER    :: pi = 5
        INTEGER,PARAMETER    :: dp = kind(1.d0)
        REAL(kind=dp)        :: r
        CHARACTER(len=pi)    :: chr1
      END MODULE dropdep_var
