  program test_a6_d3
    include "xmp_coarray.h"
    integer a2(10,1,20,0:9,21,2)[*]
    integer xmp_node_num
    integer nerr

    me = xmp_node_num()   ! == this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    if (me==1) then
       do i1=1,10
          do i2=1,1
             do i3=1,20
                do i4=0,9
                   do i5=1,21
                      do i6=1,2
                         ival=(((((i1*10+i2)*10+i3)*10+i4)*10+i5)*10+i6)
                         a2(i1,i2,i3,i4,i5,i6)=ival
                      end do
                   end do
                end do
             end do
          end do
       end do
    else
       a2=-1234
    endif
    sync all

    !---------------------------- exec
    if (me==1) then
       a2(1:10:2,:,1:10,1,1,1)[2]=a2(1:10:2,:,1:10,1,1,1)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    if (me==2) then
       do i1=1,10,2
          do i2=1,1
             do i3=1,10
                do i4=1,1
                   do i5=1,1
                      do i6=1,1
                         ival=(((((i1*10+i2)*10+i3)*10+i4)*10+i5)*10+i6)
                         jval=a2(i1,i2,i3,i4,i5,i6)
                         if (jval.ne.ival) then
                            write(*,101) i1,i2,i3,i4,i5,i6,me,jval,ival
                            nerr=nerr+1
                         end if
                         a2(i1,i2,i3,i4,i5,i6)=-1234
                      end do
                   end do
                end do
             end do
          end do
       end do

       ival=-1234
       do i1=1,10
          do i2=1,1
             do i3=1,20
                do i4=0,9
                   do i5=1,21
                      do i6=1,2
                         if (a2(i1,i2,i3,i4,i5,i6).ne.ival) then
                            write(*,101) i1,i2,i3,i4,i5,i6,me,a2(i1,i2,i3,i4,i5,i6),ival
                            nerr=nerr+1
                         end if
                      end do
                   end do
                end do
             end do
          end do
       end do
    end if

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

101 format ("a2(",i0,",",i0,",",i0,",",i0,",",i0,",",i0,")[",i0,"]=",i0," should be ",i0)

  end program
