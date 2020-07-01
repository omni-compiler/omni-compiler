  program test_a6_d4
!!     include "xmp_coarray.h"
    integer a2(10,1,20,0:9,21,2)[*]
    integer nerr

    me = this_image()

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

    !---------------------------- access coindexed variable
    if (me==1) then
       a2(:,:,1:10,:,1,1)[2]=a2(:,:,1,:,12:21,2)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    if (me==2) then
       !!! shape [10,1,10,10]
       do k1=0,9
          do k2=0,0
             do k3=0,9
                do k4=0,9
                   i1=k1+1
                   i2=k2+1
                   i3=1
                   i4=k3
                   i5=k4+12
                   i6=2
                   ival=(((((i1*10+i2)*10+i3)*10+i4)*10+i5)*10+i6)
                   lval=a2(k1+1,k2+1,k3+1,k4,1,1)
                   if (lval.ne.ival) then
                      write(*,101) k1+1,k2+1,k3+1,k4,1,1,me,lval,ival
                      nerr=nerr+1
                   end if
                   a2(k1+1,k2+1,k3+1,k4,1,1)=-1234
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
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
       call exit(1)
    end if
    !---------------------------- check and output end

101 format ("a2(",i0,",",i0,",",i0,",",i0,",",i0,",",i0,")[",i0,"]=",i0," should be ",i0)

  end program
