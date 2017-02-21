      MODULE m_procedure_declaration_7
        TYPE t
          PROCEDURE(cp),NOPASS,POINTER :: p2
        END TYPE t
      CONTAINS
        FUNCTION cp()
          TYPE(t) :: cp
        END FUNCTION cp
      end module m_procedure_declaration_7

      PROGRAM main
        USE m_procedure_declaration_7
      END PROGRAM main
