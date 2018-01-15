#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_submodule_01
        TYPE :: t
           REAL, PUBLIC :: i
           REAL, PRIVATE :: k
         CONTAINS
           PROCEDURE, PRIVATE, PASS :: p1
        END TYPE t
        COMPLEX :: i
        PRIVATE :: i, t
        INTERFACE
          MODULE FUNCTION func1(para1)
            INTEGER func1
            INTEGER para1
          END FUNCTION
        END INTERFACE
      CONTAINS
        SUBROUTINE p1(v, ret1)
          CLASS(t) :: v
          INTEGER, INTENT(OUT) :: ret1
          ret1 = v%i + v%k
        END SUBROUTINE
      END MODULE m_submodule_01

      PROGRAM main
        USE m_submodule_01
        INTEGER :: ret2
        ret2 = func1(10)
        if (ret2.eq.110) then
          PRINT *, 'PASS'
        ELSE
          PRINT *, 'NG'
          CALL EXIT(1)
        END IF
      END PROGRAM main
#else
CALL SUB1
END
#endif

