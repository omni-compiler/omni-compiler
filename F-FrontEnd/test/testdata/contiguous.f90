      PROGRAM main
        INTEGER, DIMENSION(:,:), POINTER, CONTIGUOUS :: a
        INTEGER, DIMENSION(:,:), POINTER :: b
        CONTIGUOUS :: b
       CONTAINS
        SUBROUTINE sub(c,d)
          INTEGER, DIMENSION(:), CONTIGUOUS :: c
          ! INTEGER, DIMENSION(..), CONTIGUOUS :: d
        END SUBROUTINE sub
      END PROGRAM main
