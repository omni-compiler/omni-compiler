module m_nested_common_name
contains
  subroutine sub1
    integer, save :: a0[*]
    a0 = 10
  end subroutine
  subroutine sub2
    call sub1
  end subroutine
end module

use m_nested_common_name, only : sub2

call sub1(.false.)
call sub2
call sub1(.true.)

contains
  subroutine sub1(check)
    logical check
    integer, save :: a0[*]
    if(check) then
      if(a0.eq.20) then
        print *, 'OK'
      else
        print *, 'NG : a0 = ', a0
        call exit(1)
      end if
    else
      a0 = 20
    end if
  end subroutine
end

