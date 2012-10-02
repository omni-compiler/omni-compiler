      MODULE m
      CONTAINS
        SUBROUTINE sub1(a)
          INTEGER,INTENT(INOUT) :: a
        CONTAINS
          SUBROUTINE sub2()
            INTEGER,PARAMETER :: a = 3
          END SUBROUTINE sub2
        end SUBROUTINE sub1
      END MODULE m
