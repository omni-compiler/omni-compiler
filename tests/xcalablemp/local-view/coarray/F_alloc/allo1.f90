program allo1
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:]
  real, allocatable :: b(:)

  me = this_image()
  nerr = 0

  if (allocated(a)) then
     nerr=nerr+1
     write(*,*) "1. allocated(a) must be false but:", allocated(a)
  endif

  allocate (a(10)[*])
  allocate (b(10))

  if (.not.allocated(a)) then
     nerr=nerr+1
     write(*,*) "2. allocated(a) must be true but:", allocated(a)
  endif

  deallocate (a)
  deallocate (b)

  if (allocated(a)) then
     nerr=nerr+1
     write(*,*) "3. allocated(a) must be false but:", allocated(a)
  endif

  allocate (a(10000)[*])
  allocate (b(10000))

  if (.not.allocated(a)) then
     nerr=nerr+1
     write(*,*) "4. allocated(a) must be true but:", allocated(a)
  endif

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

end program allo1

