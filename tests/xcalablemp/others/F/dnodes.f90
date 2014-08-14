program test

  include 'xmp_lib.h'

  integer size1, size2, size3
  integer result = 0;

!$xmp nodes p1(2,4)
!$xmp nodes p2(*,4)
!$xmp nodes p3(4,*)
!$xmp nodes p4(*,*,*)
!$xmp nodes p5(4,2,*)

!$xmp task on p1(1,1)
  i = xmp_nodes_size(xmp_desc_of(p1), 1, size1)
  i = xmp_nodes_size(xmp_desc_of(p1), 2, size2)

  if (size1 /= 2 .or. size2 /= 4) result = 1;

  i = xmp_nodes_size(xmp_desc_of(p2), 1, size1)
  i = xmp_nodes_size(xmp_desc_of(p2), 2, size2)

  if (size1 /= 2 .or. size2 /= 4) result = 1;

  i = xmp_nodes_size(xmp_desc_of(p3), 1, size1)
  i = xmp_nodes_size(xmp_desc_of(p3), 2, size2)

  if (size1 /= 4 .or. size2 /= 2) result = 1;

  i = xmp_nodes_size(xmp_desc_of(p4), 1, size1)
  i = xmp_nodes_size(xmp_desc_of(p4), 2, size2)
  i = xmp_nodes_size(xmp_desc_of(p4), 3, size3)

  if (size1 /= 2 .or. size2 /= 2 .or. size3 /= 2) result = 1;

  i = xmp_nodes_size(xmp_desc_of(p5), 1, size1)
  i = xmp_nodes_size(xmp_desc_of(p5), 2, size2)
  i = xmp_nodes_size(xmp_desc_of(p5), 3, size3)

  if (size1 /= 4 .or. size2 /= 2 .or. size3 /= 1) result = 1;
!$xmp end task

!$xmp task on p1(1,1)
  if (result == 0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program test
