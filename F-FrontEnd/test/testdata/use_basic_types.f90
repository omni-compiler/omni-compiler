      ! 各種基本型の参照
      PROGRAM main
        USE basic_type
        IMPLICIT NONE
        INTEGER :: int
        REAL    :: float
        LOGICAL :: bool
        int   = IABS(i)     ! check INTEGER
        int   = IFIX(r)     ! check REAL
        int   = ICHAR(chr)    ! check CHARACTER
        float = CABS(cmplx)     ! check COMPLEX
        bool  = LOGICAL(logic)  ! check LOGICAL
      END PROGRAM main
