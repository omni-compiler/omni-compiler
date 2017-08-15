      MODULE m1
        INTEGER, PUBLIC, PARAMETER :: i = 10
        INTEGER, PUBLIC, PARAMETER :: j = 10
        INTEGER, PUBLIC :: k
      END MODULE m1

      MODULE m2
        USE m1
        PRIVATE :: i
        PRIVATE :: k
      END MODULE m2

      PROGRAM main
        USE m2
        INTEGER, PARAMETER :: i = 11
        REAL :: k
      END PROGRAM main
