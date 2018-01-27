!! test for Ver.7
module m00
  integer m00(10)[*]
  me=this_image()
  ni=num_images()
contains
  subroutine ms0
    integer ms0(10)[*]
  contains
    subroutine mss
      integer mss(10)[*]
      do i=1,ni
         m00(me)[i]=me
         ms0(me)[i]=me
         mss(me)[i]=me
      enddo
      write(*,100) me,"m00",m00(1:ni)
      write(*,100) me,"ms0",ms0(1:ni)
      write(*,100) me,"mss",mss(1:ni)
      return
100   format("[",i0,"]",a,"=",(i0,","))
    end subroutine mss
  end subroutine ms0
end module m00

use m00
call mss
end
