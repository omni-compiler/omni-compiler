module parameter
      integer, parameter :: lx = @lx@
      integer, parameter :: ly = @ly@
      integer, parameter :: lz = @lz@
      integer, parameter :: lstep = @lstep@
      integer, parameter :: lnpy = @lnpy@
      integer, parameter :: lnpz = @lnpz@
!....
! !$XMP NODES proc(lnpy,lnpz) 
! !$XMP TEMPLATE t(lx,ly,lz)
! !$XMP DISTRIBUTE t(*,BLOCK,BLOCK) ONTO proc
end module parameter
