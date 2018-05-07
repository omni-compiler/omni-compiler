      MODULE tbp
        TYPE t
         CONTAINS
          PROCEDURE, NOPASS :: p => sub
        END TYPE t
       CONTAINS
        SUBROUTINE sub()
        END SUBROUTINE sub
      END MODULE tbp
