      SUBROUTINE s()
        INTERFACE
          FUNCTION x()
            COMPLEX :: x
          END FUNCTION
        END INTERFACE
       CONTAINS
        SUBROUTINE s1()
         PROCEDURE(x), POINTER :: p
        END SUBROUTINE s1
      END

      PROGRAM main
      INTERFACE
        FUNCTION x()
        END FUNCTION
      END INTERFACE
      END
