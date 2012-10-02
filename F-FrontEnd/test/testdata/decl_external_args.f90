module test
  implicit none
contains
    subroutine call_call_f()
        external funcA
        call call_f(funcA)
    end subroutine call_call_f

    subroutine call_f(external_f)
      external   external_f
      call external_f(10)
    end subroutine call_f
end module test
