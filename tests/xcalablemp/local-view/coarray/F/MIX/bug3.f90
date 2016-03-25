  include 'xmp_lib.h'

!$xmp nodes p(8)
!!$xmp nodes p4(6)=p(2:7)
!$xmp nodes p3(3,2)=p(2:7)

  me = xmp_node_num()

!$xmp task on p3
  me1 = xmp_node_num()
  print *, me, me1
!$xmp end task



  end
