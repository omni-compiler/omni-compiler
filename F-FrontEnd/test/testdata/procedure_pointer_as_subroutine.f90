      SUBROUTINE s(p1,p2)
        INTERFACE
          SUBROUTINE t
          END SUBROUTINE
        END INTERFACE
        PROCEDURE(t), POINTER, INTENT(IN)::p1
        PROCEDURE(t), POINTER, INTENT(OUT)::p2
        p2 => p1
        CALL p1
      END SUBROUTINE
