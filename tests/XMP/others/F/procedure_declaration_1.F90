#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      CALL XXX(PSS)
      CONTAINS
      SUBROUTINE XXX(PSI)
          PROCEDURE (INTEGER) :: PSI
          REAL  Y1
          Y1 = PSI()
          if(Y1.eq.10) THEN
            PRINT *, 'PASS'
          else
            PRINT *, 'NG'
            CALL EXIT(1)
          end if
      END SUBROUTINE
      FUNCTION PSS()
          INTEGER :: PSS
          PSS = 10
      END FUNCTION
      END

#else
PRINT *, 'SKIPPED'
END
#endif
