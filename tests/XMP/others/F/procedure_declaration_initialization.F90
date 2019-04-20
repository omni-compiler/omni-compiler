#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      MODULE m_procedure_declaration_initialization
        public :: h
        INTERFACE
           FUNCTION f(a)
             INTEGER :: f
             INTEGER :: a
           END FUNCTION f
        END INTERFACE
        PROCEDURE(f),POINTER :: g => h
      CONTAINS
        FUNCTION h(a)
          INTEGER :: h
          INTEGER :: a
          h = a + 10
        END FUNCTION h
      END MODULE

      use m_procedure_declaration_initialization

      integer ret

      ret = g(20)

      if(ret.eq.30) then
        PRINT *, 'PASS'
      else
        PRINT *, 'NG'
        CALL EXIT(1)
      end if

      END
#else
PRINT *, 'SKIPPED'
END
#endif
