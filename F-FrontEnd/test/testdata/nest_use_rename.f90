      PROGRAM main
        USE use_rename_vars, p => s, q => t
        IMPLICIT NONE
        REAL(kind=p) :: r
        LOGICAL :: a = q
        CHARACTER :: b
        COMPLEX :: s
        INTEGER :: t
      END PROGRAM main

