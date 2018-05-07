program tgmove
integer :: i,irank,xmp_node_num
integer,parameter :: n=8
integer a(n,n,n,n,n,n,n),b(n,n,n,n,n,n,n)
!$xmp nodes p(2,2,2,2,2,2,2)
!$xmp template tx(n,n,n,n,n,n,n)
!$xmp distribute tx(block,block,block,block,block,block,block) onto p
!$xmp align a(i0,i1,i2,i3,i4,i5,i6) with tx(i0,i1,i2,i3,i4,i5,i6)
!$xmp align b(i0,*,i2,*,i4,*,i6) with tx(i0,*,i2,*,i4,*,i6)

irank=xmp_node_num()

!$xmp loop (i0,i1,i2,i3,i4,i5,i6) on tx(i0,i1,i2,i3,i4,i5,i6)
do i6=1,n
  do i5=1,n
    do i4=1,n
      do i3=1,n
        do i2=1,n
          do i1=1,n
            do i0=1,n
              a(i0,i1,i2,i3,i4,i5,i6)=i0+i1+i2+i3+i4+i5+i6
            end do
          end do
        end do
      end do
    end do
  end do
end do

!$xmp loop (i0,i2,i4,i6) on tx(i0,*,i2,*,i4,*,i6)
do i6=1,n
  do i5=1,n
    do i4=1,n
      do i3=1,n
        do i2=1,n
          do i1=1,n
            do i0=1,n
              b(i0,i1,i2,i3,i4,i5,i6)=0
            end do
          end do
        end do
      end do
    end do
  end do
end do

!$xmp gmove
b(2:n,2:n,2:n,2:n,2:n,2:n,2:n)=a(2:n,2:n,2:n,2:n,2:n,2:n,2:n)

ierr=0
!$xmp loop (i0,i2,i4,i6) on tx(i0,*,i2,*,i4,*,i6)
do i6=2,n
  do i5=2,n
    do i4=2,n
      do i3=2,n
        do i2=2,n
          do i1=2,n
            do i0=2,n
              ierr=ierr+abs(b(i0,i1,i2,i3,i4,i5,i6)-i0-i1-i2-i3-i4-i5-i6)
            end do
          end do
        end do
      end do
    end do
  end do
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)

stop
end program
