       INTERFACE
         SUBROUTINE sub()
         END SUBROUTINE sub
       END INTERFACE
       TYPE t
         PROCEDURE(sub), POINTER, NOPASS :: proc => sub
       END TYPE t

       TYPE(t) :: u = t()
       CALL u%proc()
       END
