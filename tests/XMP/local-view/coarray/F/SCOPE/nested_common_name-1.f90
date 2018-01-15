module m_nested_common_name_1
  integer a0[*]
  integer a1[*]
  integer a2[*]
contains
  subroutine sub1(v)
    integer v
    integer, save :: a0[*]
    integer, save :: a3[*]
    a0 = v
    a1 = v
    a2 = v
    a3 = v
  end subroutine
end module

use m_nested_common_name_1

integer a3[*]

a0 = 10
a3 = 10000
call sub3(.false.)
call sub1(100)
if(a0.eq.10.and.a1.eq.100.and.a3.eq.10000) then
  print *, 'OK 1'
else
  print *, 'NG 1 : a0 = ', a0, 'a1 = ', a1
end if
call sub2
call sub3(.true.)

contains
  subroutine sub2
    if(a0.eq.10.and.a1.eq.100.and.a3.eq.10000) then
      print *, 'OK 2'
    else
      print *, 'NG 2 : a0 = ', a0, 'a1 = ', a1
    end if
  end subroutine
  subroutine sub3(check)
    logical check
    integer, save :: a2[*]
    if (.not.check) then
      a2 = 1000
    else
      if(a2.eq.1000) then
        print *, 'OK 3'
      else
        print *, 'NG 3 : a2 = ', a2
        call exit(1)
      end if
    end if
  end subroutine
end

