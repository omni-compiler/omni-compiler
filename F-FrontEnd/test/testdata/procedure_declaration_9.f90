      SUBROUTINE s()
        INTERFACE
          FUNCTION x()
          END FUNCTION
        END INTERFACE
        PROCEDURE(x), POINTER :: p
      END

      INTERFACE
        FUNCTION x()
        END FUNCTION
      END INTERFACE
      END
