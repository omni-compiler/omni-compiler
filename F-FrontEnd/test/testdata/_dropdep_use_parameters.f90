      MODULE dropdep_use_parameters
        USE parameters
        IMPLICIT NONE
        PRIVATE
        PUBLIC :: r, chr1
        REAL(kind=dp)        :: r
        CHARACTER(len=pi)    :: chr1
      END MODULE dropdep_use_parameters
