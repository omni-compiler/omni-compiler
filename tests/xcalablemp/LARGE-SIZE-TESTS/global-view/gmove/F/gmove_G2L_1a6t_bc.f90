program tgmove
integer,parameter :: n=4
integer a(n,n,n,n,n,n), b(n,n,n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2,2,2,2)
!$xmp template tx(n,n,n,n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2),cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(*,*,*,*,*,i5) with tx(*,*,*,*,*,i5)

!$xmp loop (i5) on tx(*,*,*,*,*,i5)
do i5=1,n
  do i4=1,n
    do i3=1,n
      do i2=1,n
        do i1=1,n
          do i0=1,n
            a(i0,i1,i2,i3,i4,i5)=i0+i1+i2+i3+i4+i5
          end do
        end do
      end do
    end do
  end do
end do

do i5=1,n
  do i4=1,n
    do i3=1,n
      do i2=1,n
        do i1=1,n
          do i0=1,n
            b(i0,i1,i2,i3,i4,i5)=0
          end do
        end do
      end do
    end do
  end do
end do

!$xmp gmove
b(2:n,2:n,2:n,2:n,2:n,2:n)=a(2:n,2:n,2:n,2:n,2:n,2:n)

ierr=0
do i5=2,n
  do i4=2,n
    do i3=2,n
      do i2=2,n
        do i1=2,n
          do i0=2,n
            ierr=ierr+abs(b(i0,i1,i2,i3,i4,i5)-i0-i1-i2-i3-i4-i5)
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
