#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_procedure_declaration_7
        TYPE t
          PROCEDURE(cp1),NOPASS,POINTER :: p => null()
        END TYPE t
        INTEGER :: CNT = 0
      CONTAINS
        FUNCTION cp1(o1)
          TYPE(t) :: cp1
          TYPE(t) :: o1
          CNT = CNT + 1
          cp1 = o1
        END FUNCTION cp1
        FUNCTION cp2(o2)
          TYPE(t) :: cp2
          TYPE(t) :: o2
          CNT = CNT + 2
          cp2 = o2
        END FUNCTION cp2
      end module m_procedure_declaration_7

      PROGRAM main
        USE m_procedure_declaration_7
        TYPE(t) :: o3
        o3%p => cp1
        o3 = o3%p(o3)
        o3%p => cp2
        o3 = o3%p(o3)
        IF (CNT.EQ.3) THEN
          PRINT *, 'PASS'
        ELSE
          PRINT *, 'NG'
          CALL EXIT(1)
        END IF
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
