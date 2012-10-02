      ! 定数リテラルを用いて作られた型の参照
      PROGRAM main
        USE derivered_types_with_literal
        IMPLICIT NONE
        REAL*8    :: double
        double = DABS(r8)
      END PROGRAM main
