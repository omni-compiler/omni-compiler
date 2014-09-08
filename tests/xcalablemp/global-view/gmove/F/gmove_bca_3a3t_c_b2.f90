program tgmove
integer,parameter :: n=16
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,3,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp distribute ty(block,block,block) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i2,i1) with ty(*,i1,i2)

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i2,i1)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i2,i1)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
call chk_int(ierr)

stop
end program tgmove
