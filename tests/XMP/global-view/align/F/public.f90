module mmm

  !$xmp nodes p(2)
  !$xmp template t(100)
  !$xmp distribute t(block) onto p
  
  real(8), public :: aaa(100)
  !$xmp align aaa(i) with t(i)

end module mmm
