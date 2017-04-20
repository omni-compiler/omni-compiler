      PROGRAM main
        USE use_procedure_decls

        INTERFACE
          FUNCTION u(arg)
            INTEGER :: u
            INTEGER :: arg
          END FUNCTION u
          SUBROUTINE v()
          END SUBROUTINE v
        END INTERFACE

        a => u
        b => v
      END PROGRAM main
