module parameter
      integer, parameter :: lx = @lx@
      integer, parameter :: ly = @ly@
      integer, parameter :: lz = @lz@
      integer, parameter :: lstep = @lstep@
      integer, parameter :: lnpz = @lnpz@
!....
! !$XMP NODES proc(lnpz) 
! !$XMP TEMPLATE t(lx,ly,lz)
! !$XMP DISTRIBUTE t(*,*,BLOCK) ONTO proc
end module parameter
