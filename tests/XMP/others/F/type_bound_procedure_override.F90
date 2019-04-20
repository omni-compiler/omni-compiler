#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_type_bound_procedure_override
        TYPE t
         CONTAINS
           PROCEDURE, PUBLIC :: f
        END TYPE t
        TYPE, EXTENDS(t) :: tt
         CONTAINS
           PROCEDURE, PUBLIC :: f => ff
        END TYPE tt
      CONTAINS
        FUNCTION f(this, i)
          INTEGER :: f
          CLASS(t) :: this
          INTEGER :: i
          f = i + 2
!         PRINT *, "call F"
        END FUNCTION f
        FUNCTION ff(this, i)
          INTEGER :: ff
          CLASS(tt) :: this
          INTEGER :: i
          ff = i + 3
!         PRINT *, "call FF"
        END FUNCTION ff
      END MODULE m_type_bound_procedure_override

      PROGRAM main
        USE m_type_bound_procedure_override
        integer ans
        type(t) o1
        type(tt) o2
        ans = 1
        ans = o1%f(ans)
        ans = o2%f(ans)
        if (ans.eq.6) then
          print *, 'PASS'
        else
          print *, 'NG : ', ans, ', should be 6'
          call exit(1)
        endif
      END PROGRAM main
#else
print *, 'SKIPPED'
end
#endif
