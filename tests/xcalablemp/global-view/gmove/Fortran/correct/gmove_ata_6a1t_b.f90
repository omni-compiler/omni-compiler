program tgmove
integer i0,i1,i2,i3,i4,i5,n=4
integer a(n,n,n,n,n,n),b(n,n,n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(*,*,*,*,*,i) with tx(i)
!$xmp align b(*,i,*,*,*,*) with tx(i)

!$xmp loop (i5) on tx(i5)
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
!$xmp loop (i1) on tx(i1)
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
b(1:n,1:n,1:n,1:n,1:n,1:n)=a(1:n,1:n,1:n,1:n,1:n,1:n)

ierr=0
!$xmp loop (i5) on tx(i5)
do i5=1,n
  do i4=1,n
    do i3=1,n
      do i2=1,n
!$xmp loop (i1) on tx(i1)
        do i1=1,n
          do i0=1,n
            ierr=ierr+abs(b(i0,i1,i2,i3,i4,i5)-a(i0,i1,i2,i3,i4,i5))
          end do
        end do
      end do
    end do
  end do
end do

!$xmp reduction (+:ierr)
irank=xmp_node_num()
if (irank==1) then
  print *, 'max error=',ierr,'irank=',irank
endif

contains
end
