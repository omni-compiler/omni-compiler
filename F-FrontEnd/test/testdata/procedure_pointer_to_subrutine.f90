      MODULE m
        TYPE t
          PROCEDURE(sub2), POINTER, NOPASS :: p => NULL()
        END TYPE t
       CONTAINS
        SUBROUTINE sub1()
          TYPE(t) :: v
          v%p => sub2
          CALL v%p()
        END SUBROUTINE sub1
        SUBROUTINE sub2()
        END SUBROUTINE sub2
      END MODULE m
 
        
