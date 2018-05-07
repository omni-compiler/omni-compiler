module m_nested_common_name_1_0_0
  integer a0[*]
contains
  subroutine sub1(v)
    integer v
    a0 = v
  end subroutine
end module

use m_nested_common_name_1_0_0

a0 = 10
call sub1(100)
if(a0.eq.100) then
  print *, 'OK 1'
else
  print *, 'NG 1 : a0 = ', a0
end if
a0 = 1000
call sub2

contains
  subroutine sub2
    if(a0.eq.1000) then
      print *, 'OK 2'
    else
      print *, 'NG 2 : a0 = ', a0
    end if
  end subroutine
end

