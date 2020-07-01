program tgmove
integer i
integer,parameter :: n=8
integer a(n,n,n,n,n,n,n),b(n,n,n,n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2,2,2,2,2)
!$xmp template tx(n,n,n,n,n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2),cyclic(2),cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i0,i1,i2,i3,i4,i5,i6) with tx(i0,i1,i2,i3,i4,i5,i6)

!$xmp loop on tx(i0,i1,i2,i3,i4,i5,i6)
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
b(2:5,2:5,2:5,2:5,2:5,2:5,2:5)=a(5:8,5:8,5:8,5:8,5:8,5:8,5:8)

ierr=0
do i6=2,5
  do i5=2,5
    do i4=2,5
      do i3=2,5
        do i2=2,5
          do i1=2,5
            do i0=2,5
              ierr=ierr+abs(b(i0,i1,i2,i3,i4,i5,i6)-(i0+3)-(i1+3)-(i2+3)-(i3+3)-(i4+3)-(i5+3)-(i6+3))
            end do
          end do
        end do
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
call chk_int(ierr)

stop
end program tgmove
