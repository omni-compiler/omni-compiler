program gmove_ata_4

  character*25 tname
  call gmove_ata_2a1t_subcomm(tname)

end program

subroutine gmove_ata_2a1t_subcomm(tname)

character(*) tname
integer :: i,j
integer,parameter :: n=5
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(4)
!$xmp nodes p1(2)=p(1:2)
!$xmp nodes p2(2)=p(3:4)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p1
!$xmp distribute ty(block) onto p2
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,i) with ty(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (j) on ty(j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i) on ty(i)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(j,i)-i-j)
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_2a1t_subcomm"
call chk_int(tname, ierr)

end subroutine

