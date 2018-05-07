  subroutine sub
!!     include 'xmp_coarray.h'
    real a
    n=this_image()
    write(*,*) a
    entry sub1
    write(*,*) a
  end subroutine sub
