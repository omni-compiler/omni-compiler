module bb1
  !$xmp nodes p(3)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p

contains
  subroutine bb1m
    real a(10,10)
    !$xmp align a(*,i) with t(i)
  end subroutine bb1m
end module bb1


use bb1
call bb1m
print *,"OK"
end

