      MODULE m_procedure_declaration_4_1

        INTERFACE
          MODULE FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE

        PROCEDURE(f), POINTER :: g2 => f
        PROCEDURE(g2), POINTER :: g10

      CONTAINS

        MODULE PROCEDURE f
          f = a - 1
        END PROCEDURE f

      end module m_procedure_declaration_4_1

      PROGRAM main
        USE m_procedure_declaration_4_1
      END PROGRAM main
