#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_type_bound_procedure_3
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
           PROCEDURE, PUBLIC :: g => f
        END TYPE t
      CONTAINS
        FUNCTION f(this, w) RESULT(ans)
          IMPLICIT NONE
          CLASS(t) :: this
          INTEGER :: w
          INTEGER :: ans
          ans = this%v + w
        END FUNCTION f
      END MODULE m_type_bound_procedure_3

      PROGRAM main
        USE m_type_bound_procedure_3

        type(t) o1
        o1 = t(10)
!       print *, o1%g(2)
        if (o1%g(2).eq.12) then
          print *, 'PASS'
        else
          print *, 'NG'
          call exit(1)
        endif

      END PROGRAM main
#else
print *, 'SKIPPED'
end
#endif
