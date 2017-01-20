      MODULE procedure_decls
        PRIVATE :: f, sub
        PROCEDURE(f), POINTER :: a => f
        PROCEDURE(sub), POINTER :: b => sub
       CONTAINS
        FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
        END FUNCTION f
        SUBROUTINE sub()
        END SUBROUTINE sub
      END MODULE procedure_decls

      MODULE m
        USE procedure_decls
      END MODULE m
