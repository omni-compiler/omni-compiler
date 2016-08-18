
      PROGRAM MAIN
      CONTAINS
        SUBROUTINE sub()

1000      WRITE (*,*) "hoge"
          BLOCK
            GOTO 1000
          END BLOCK
        END SUBROUTINE sub

      END PROGRAM MAIN
