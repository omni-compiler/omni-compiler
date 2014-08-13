program tgmove
integer,parameter :: n=4
integer a(n,n,n), b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(*,*,i2) with tx(*,*,i2)

!$xmp loop (i2) on tx(*,*,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i1,i2)=0
    end do
  end do
end do

!$xmp gmove
b(:,:,:)=a(:,:,:)

ierr=0
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i1,i2)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
irank=xmp_node_num()
if (irank==1) then
  print *, 'max error=',ierr
endif
!call exit(ierr)

stop
end program tgmove
