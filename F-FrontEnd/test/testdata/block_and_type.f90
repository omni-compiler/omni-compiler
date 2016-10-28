      PROGRAM MAIN
        TYPE tt
           INTEGER :: b
        END type tt
        TYPE(tt) :: c
        BLOCK
          TYPE tt
             INTEGER :: a
          END type tt
          TYPE(tt) :: b
        END BLOCK
      END PROGRAM MAIN
