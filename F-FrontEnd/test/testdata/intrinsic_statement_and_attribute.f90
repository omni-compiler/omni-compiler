      PROGRAM main
       CONTAINS
        SUBROUTINE case1()
          ! case 1
          REAL, INTRINSIC :: SQRT
        END SUBROUTINE case1
        SUBROUTINE case2()
          ! case 2
          COMPLEX, INTRINSIC :: SQRT
        END SUBROUTINE case2
        SUBROUTINE case3()
          ! case 3
          INTEGER, INTRINSIC :: RESHAPE
        END SUBROUTINE case3
        SUBROUTINE case4()
          ! case 4
          INTEGER, INTRINSIC :: SQRT
        END SUBROUTINE case4
        SUBROUTINE case5()
          ! case 5
          COMPLEX, INTRINSIC :: CSQRT
        END SUBROUTINE case5
        SUBROUTINE case6()
          ! case 6
          REAL(8), INTRINSIC :: CSQRT
        END SUBROUTINE case6
      END PROGRAM main
