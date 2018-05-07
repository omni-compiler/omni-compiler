!-----------------------------------------------------------------------
!                            /      /     /
      subroutine measure ( kmode, ename, kid )
!
!
!   <arguments>
!     kmode : mode 0 = print measurement
!           : mode 1 = initialize for kid
!           : mode 2 = start measurement for kid
!           : mode 3 = end measurement for kid
!
!   <remarks>
!     none
!
!    coded by Sakagami,H. (NIFS) 13/10/18
!-----------------------------------------------------------------------
      integer, parameter :: lnum = 10
      character*16 :: cname(lnum)
      character*(*) :: ename
      real*8 :: asum(lnum), ats(lnum) 
      data cname / lnum * '                ' /
      data asum / lnum * -999999.9d0 /
      real*8 :: fgetwtod
      save
!....
      if( kmode .eq. 0 ) then
         write(*,*) '--------------------------------------------------'
         write(*,*) 'measured time (sec)'
	 do i = 1, lnum
	    if( asum(i) .ge. 0.0d0 ) write(*,*) cname(i), asum(i)
	 end do
         write(*,*) '--------------------------------------------------'
      else if( kmode .eq. 1 ) then
         asum(kid) = 0.0d0
	 cname(kid) = ename
      else if( kmode .eq. 2 ) then
         ats(kid) = fgetwtod()
      else if( kmode .eq. 3 ) then
	 asum(kid) = asum(kid) + ( fgetwtod() - ats(kid) )
      else
         write(*,*) '### ERROR : kid = ', kid
      end if
!....
      return
      end
