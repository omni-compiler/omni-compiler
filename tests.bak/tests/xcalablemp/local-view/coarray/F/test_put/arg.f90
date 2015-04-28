  program put_a2d1_NG
    include "xmp_coarray.h"
    real a2(2,3)[*]
    call realsub(a1)
  end program

  subroutine realsub(a2)
    include "xmp_coarray.h"
    real a2(2,3)[*]
    integer xmp_node_num, xmp_num_nodes
    integer nerr

    !---------------------------- check and preparation
    me = xmp_node_num()   ! == this_image()

    if (xmp_num_nodes() < 3) then
       print '("[",i0,"] stop: at least 3 images are needed")', me
       stop
    end if

    !---------------------------- init
    a2=7.77
    sync all

    !---------------------------- exec
    if (me==1) then
       a2(1:2,2)[2]=(/1.22,2.22/)
    else if (me==2) then
       a2(2,1:3:2)[3]=(/2.13,2.33/)
    end if

    sync all
    !---------------------------- check and output start
    eps = 0.00001
    nerr = 0
    do j=1,3
       do i=1,2
          if (i==1.and.j==2.and.me==2) then
             val=1.22
             if (abs(a2(i,j)-val)>eps) then
                write(*,100) i,j,me,a2(i,j),val
                nerr=nerr+1
             end if
          else if (i==2.and.j==2.and.me==2) then
             val=2.22
             if (abs(a2(i,j)-val)>eps) then
                write(*,100) i,j,me,a2(i,j),val
                nerr=nerr+1
             end if
          else if (i==2.and.j==1.and.me==3) then
             val=2.13
             if (abs(a2(i,j)-val)>eps) then
                write(*,100) i,j,me,a2(i,j),val
                nerr=nerr+1
             end if
          else if (i==2.and.j==3.and.me==3) then
             val=2.33
             if (abs(a2(i,j)-val)>eps) then
                write(*,100) i,j,me,a2(i,j),val
                nerr=nerr+1
             end if
          else
             val=7.77
             if (abs(a2(i,j)-val)>eps) then
                write(*,100) i,j,me,a2(i,j),val
                nerr=nerr+1
             end if
          end if
       end do
    end do

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

100 format ("a2(",i0,",",i0,")[",i0,"]=",f8.6," should be ",f8.6)

  end
