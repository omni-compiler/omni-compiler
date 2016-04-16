PROGRAM zzz
  include 'xmp_lib.h'
!!   include 'xmp_coarray.h'

!$xmp nodes p(3)
!$xmp nodes pp(2)=p(2:3)

      write(*,*) "out:", this_image(), xmp_node_num()

!$xmp task on pp
      write(*,*) "in: ", this_image(), xmp_node_num()

!$xmp end task

      END
