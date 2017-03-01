#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_type_bound_procedure_4
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
        END TYPE t
      CONTAINS
        FUNCTION f(this)
          IMPLICIT NONE
          CLASS(t) :: this
          TYPE(t) :: f
          f = t(this%v+10)
        END FUNCTION f
      END MODULE m_type_bound_procedure_4

      PROGRAM main
        USE m_type_bound_procedure_4
        TYPE(t), TARGET :: a
        TYPE(t)  :: b
        CLASS(t), POINTER :: p
        a = t(1)
        p => a
        b = p%f()
!       print *, b
        if (b%v.eq.11) then
          print *, 'PASS'
        else
          print *, 'NG : ', b%v, ', should be 11'
          call exit(1)
        endif
      END PROGRAM main
#else
print *, 'SKIPPED'
end
#endif
