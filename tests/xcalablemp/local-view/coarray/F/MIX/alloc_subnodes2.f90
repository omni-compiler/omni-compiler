  program times2
!$xmp nodes p(8)

    real, allocatable :: a(:,:)[:]
!!!$xmp coarray on p :: a
    real, allocatable :: b(:,:)[:]
    real, allocatable :: c(:,:)[:]
    real, allocatable :: d(:,:)[:]
    me = this_image()

    tmp=1234.567
    do i=1,20
       allocate(a(1000,569)[*],b(1000,800)[*])
       b(32,55)=tmp*me
       sync all
       a(44,44)=b(32,55)[7]/7
       sync all
       deallocate(b)
       n1=mod(i,5)+1
       n2=n1+mod(i,3)+1
       if (me/=n1) a(44,44)=huge(a)
!$xmp task on p(n1:n2)
       !! at least 2 nodes execute.
       allocate(c(1300,1000)[*],d(1001,1001)[*])
       if (this_image()==2) then
!!!!!!!!!!!!!!!!!
!!          c(1000,i)=a(44,44)[n1]
!!!!!!!!!!!!!!!!!
          c(1000,i)=a(44,44)[1]
       endif
       sync all
       a(1,1)=c(1000,i)[2]
       deallocate(c,d)
!$xmp end task
       sync all
       tmp=a(1,1)[n2]
       deallocate(a)
    end do

    nerr = 0
    na = xmpf_coarray_allocated_bytes()
    ng = xmpf_coarray_garbage_bytes()
    if (na /= 0 .or. ng /= 0) then 
       nerr = nerr+1
       write(*,100) this_image(), na, ng
    endif
    if (abs(tmp-1234.567)>0.0001) then
       nerr = nerr+1
       write(*,110) this_image(), 1234.567, tmp
    endif

    call final_msg(na)

100 format("[",i0,"] remains allocated ",i0," and gabage ",i0," bytes")
110 format("[",i0,"] tmp should be ",f10.3," but ",f10.3)

  end program times2


  subroutine final_msg(nerr)
!!     include 'xmp_coarray.h'
    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if
    return
  end subroutine final_msg

