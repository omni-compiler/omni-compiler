program test
  implicit none
  type mytype
     integer:: a
  end type mytype
  type(mytype):: mt
  !$xmp nodes p(*)
  mt%a = 2
  !$xmp bcast (mt%a)
  print *, mt%a
end program test
