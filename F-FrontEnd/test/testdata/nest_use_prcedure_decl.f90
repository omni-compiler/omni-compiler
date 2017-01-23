      PROGRAM main
        USE use_procedure_decls

        INTERFACE 
          FUNCTION u(a)
            INTEGER :: u
            INTEGER :: a
          END FUNCTION u
          SUBROUTINE v()
          END SUBROUTINE v
        END INTERFACE

        a => u
        b => v
      END PROGRAM main
