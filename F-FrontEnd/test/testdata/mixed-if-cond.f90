  subroutine x()
    real*8, allocatable :: a(:)
    if(ihi==p-1.and.maxval(abs(a))==0) then
       print *, "hoehoe"
    endif
  end subroutine x



