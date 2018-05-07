      MODULE m
        INTEGER, PUBLIC, PROTECTED :: i
        INTEGER, PRIVATE, PROTECTED :: j
      END MODULE m

      MODULE m1
        USE m
        PUBLIC :: i
      END MODULE m1

      MODULE m2
        USE m
        PRIVATE :: i
      END MODULE m2

      PROGRAM main
        USE m1
      END PROGRAM
   
