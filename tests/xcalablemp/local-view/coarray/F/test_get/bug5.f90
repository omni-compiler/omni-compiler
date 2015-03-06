  interface foo
     integer function fooi(n)
       integer n
     end function
     real function foor(r)
       real r
     end function
  end interface

!$xmp nodes p(4)
!$xmp template t(100)
!$xmp distribute t(block) onto p

!$xmp loop on t(i+foo(0))
  do i=1,100
  enddo

  end

