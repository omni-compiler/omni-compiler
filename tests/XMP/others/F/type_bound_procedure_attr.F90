#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
               MODULE m_type_bound_procedure_attr
                 TYPE :: t
                   INTEGER :: v
                 CONTAINS
                   PROCEDURE,PASS(x),non_overridable :: show => show
                   PROCEDURE,PASS(x),private:: show1
                 END TYPE t
               CONTAINS
                 SUBROUTINE show(x, v1)
                   integer v1
                   CLASS(t) :: x
                   call show1(x, v1)
                 END SUBROUTINE show
                 SUBROUTINE show1(x, v1)
                   integer v1
                   CLASS(t) :: x
!                  PRINT *, v1 + x%v
                   if (v1+x%v.eq.3) then
                     print *, 'PASS'
                   else
                     print *, 'NG : ', v1+x%v, ', should be 3'
                     call exit(1)
                   endif
                 END SUBROUTINE show1
               END MODULE m_type_bound_procedure_attr

               PROGRAM MAIN
                 USE m_type_bound_procedure_attr
                 CLASS(t), POINTER :: a
                 TYPE(t), TARGET :: b = t(v=1)
                 a => b
                 CALL a%show(2)
               END PROGRAM MAIN
#else
print *, 'SKIPPED'
end
#endif
