      SUBROUTINE sub(a)
        INTEGER, CODIMENSION[*], VOLATILE :: a
      END SUBROUTINE sub

      PROGRAM MAIN
        INTERFACE
          SUBROUTINE sub(a)
            INTEGER, CODIMENSION[*], VOLATILE :: a
          END SUBROUTINE sub
        END INTERFACE
        INTEGER, DIMENSION(3,3), CODIMENSION[2,*] :: a
        CALL sub(a(1,1))
      END PROGRAM MAIN
