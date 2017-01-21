      MODULE use_procedure_decls
        USE procedure_decls
       CONTAINS
        SUBROUTINE check()
          integer :: i
          a => p
          b => q
          i = p (1)
        END SUBROUTINE check
        FUNCTION p(arg)
          INTEGER :: p
          INTEGER :: arg
        END FUNCTION p
        SUBROUTINE q()
        END SUBROUTINE q
      END MODULE use_procedure_decls
