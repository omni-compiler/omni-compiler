module parameter
      integer, parameter :: lx = 128
      integer, parameter :: ly = 128
      integer, parameter :: lz = 128
      integer, parameter :: lstep = 10
      integer, parameter :: lnpx = 2
      integer, parameter :: lnpy = 2
      integer, parameter :: lnpz = 2
!....
! !$XMP NODES proc(lnpx,lnpy,lnpz) 
! !$XMP TEMPLATE t(lx,ly,lz)
! !$XMP DISTRIBUTE t(BLOCK,BLOCK,BLOCK) ONTO proc
end module parameter
