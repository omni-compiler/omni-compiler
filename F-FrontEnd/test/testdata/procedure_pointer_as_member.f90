       INTERFACE
         SUBROUTINE sub()
         END SUBROUTINE sub
       END INTERFACE
       TYPE t
         PROCEDURE(sub), POINTER, NOPASS :: proc => NULL()
       END TYPE t

       TYPE(t) :: u = t()
       u%proc => sub
       CALL u%proc()
       END
