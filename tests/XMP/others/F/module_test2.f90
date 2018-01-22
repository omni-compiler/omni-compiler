module second
!$xmp nodes p(2)
!$xmp template t(10)
!$xmp distribute t(block) onto p

integer a(10)
!$xmp align a(i) with t(i)
contains
  subroutine foo()
    integer ::i
    !$xmp loop on t(i)
    do i=1, 10
       a(i) = i
    end do

    !$xmp barrier

    write(*,*) "PASS"
  end subroutine foo
end module second
