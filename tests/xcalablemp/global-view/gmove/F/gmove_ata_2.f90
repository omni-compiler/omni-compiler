program gmove_ata_2

  character*25 tname
  call gmove_ata_2a1t_b(tname)
  call gmove_ata_2a1t_b_h(tname)
  call gmove_ata_2a1t_c(tname)
  call gmove_ata_3a1t_b2(tname)
  call gmove_ata_3a1t_b(tname)
  call gmove_ata_4a1t_b(tname)
  call gmove_ata_5a1t_b(tname)
  call gmove_ata_6a1t_b(tname)
  call gmove_ata_7a1t_b_c(tname)
  call gmove_ata_7a1t_b(tname)
  call gmove_ata_7a1t_c(tname)

end program


subroutine gmove_ata_2a1t_b(tname)

character(*) tname
integer :: i,j
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,i) with tx(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (j) on tx(j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (j) on tx(j)
do j=1,n
!$xmp loop (i) on tx(i)
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_2a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_2a1t_b_h(tname)

character(*) tname
integer :: i,j
integer,parameter :: n=5
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,i) with tx(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (j) on tx(j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
do j=1,n
!$xmp loop (i) on tx(i)
  do i=1,n
    ierr=ierr+abs(b(j,i)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_2a1t_b_h"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_2a1t_c(tname)

character(*) tname
integer :: i,j
integer,parameter ::n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(cyclic) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,i) with tx(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (j) on tx(j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (j) on tx(j)
do j=1,n
!$xmp loop (i) on tx(i)
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_2a1t_c"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_3a1t_b2(tname)

character(*) tname
integer :: i0,i1,i2
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i,*,*) with tx(i)
!$xmp align b(*,i,*) with tx(i)

!$xmp loop (i0) on tx(i0)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

do i2=1,n
!$xmp loop (i1) on tx(i1)
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
!$xmp loop (i1) on tx(i1)
  do i1=1,n
!$xmp loop (i0) on tx(i0)
    do i0=1,n
      ierr=ierr+abs(b(i0,i1,i2)-a(i0,i1,i2))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_3a1t_b2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_3a1t_b(tname)

character(*) tname
integer :: i0,i1,i2
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i,*,*) with tx(i)
!$xmp align b(*,i,*) with tx(i)

!$xmp loop (i0) on tx(i0)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

do i2=1,n
!$xmp loop (i1) on tx(i1)
  do i1=1,n
    do i0=1,n
      b(i0,i1,i2)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
do i2=1,n
!$xmp loop (i1) on tx(i1)
  do i1=1,n
!$xmp loop (i0) on tx(i0)
    do i0=1,n
      ierr=ierr+abs(b(i0,i1,i2)-a(i0,i1,i2))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_3a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_4a1t_b(tname)

character(*) tname
integer i0,i1,i2,i3
integer,parameter :: n=4
integer a(n,n,n,n),b(n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(*,*,*,i) with tx(i)
!$xmp align b(*,i,*,*) with tx(i)

!$xmp loop (i3) on tx(i3)
do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        a(i0,i1,i2,i3)=i0+i1+i2+i3
      end do
    end do
  end do
end do

do i3=1,n
  do i2=1,n
!$xmp loop (i1) on tx(i1)
    do i1=1,n
      do i0=1,n
        b(i0,i1,i2,i3)=0
      end do
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n,1:n)=a(1:n,1:n,1:n,1:n)

ierr=0
!$xmp loop (i3) on tx(i3)
do i3=1,n
  do i2=1,n
!$xmp loop (i1) on tx(i1)
    do i1=1,n
      do i0=1,n
        ierr=ierr+abs(b(i0,i1,i2,i3)-a(i0,i1,i2,i3))
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_4a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_5a1t_b(tname)

character(*) tname
integer i0,i1,i2,i3,i4
integer,parameter :: n=4
integer a(n,n,n,n,n),b(n,n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(*,*,*,*,i) with tx(i)
!$xmp align b(*,i,*,*,*) with tx(i)

!$xmp loop (i4) on tx(i4)
do i4=1,n
  do i3=1,n
    do i2=1,n
      do i1=1,n
        do i0=1,n
          a(i0,i1,i2,i3,i4)=i0+i1+i2+i3+i4
        end do
      end do
    end do
  end do
end do

do i4=1,n
  do i3=1,n
    do i2=1,n
!$xmp loop (i1) on tx(i1)
      do i1=1,n
        do i0=1,n
          b(i0,i1,i2,i3,i4)=0
        end do
      end do
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n,1:n,1:n)=a(1:n,1:n,1:n,1:n,1:n)

ierr=0
!$xmp loop (i4) on tx(i4)
do i4=1,n
  do i3=1,n
    do i2=1,n
!$xmp loop (i1) on tx(i1)
      do i1=1,n
        do i0=1,n
          ierr=ierr+abs(b(i0,i1,i2,i3,i4)-a(i0,i1,i2,i3,i4))
        end do
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_5a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_6a1t_b(tname)

character(*) tname
integer i0,i1,i2,i3,i4,i5
integer,parameter :: n=4
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

!$xmp reduction (max:ierr)
tname="gmove_ata_6a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_7a1t_b_c(tname)

character(*) tname
integer i0,i1,i2,i3,i4,i5,i6
integer,parameter :: n=4
integer a(n,n,n,n,n,n,n),b(n,n,n,n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp align a(*,*,*,*,*,*,i) with tx(i)
!$xmp align b(*,i,*,*,*,*,*) with ty(i)

!$xmp loop (i6) on tx(i6)
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
!$xmp loop (i1) on ty(i1)
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
b(1:n,1:n,1:n,1:n,1:n,1:n,1:n)=a(1:n,1:n,1:n,1:n,1:n,1:n,1:n)

ierr=0
!$xmp loop (i6) on tx(i6)
do i6=1,n
  do i5=1,n
    do i4=1,n
      do i3=1,n
        do i2=1,n
!$xmp loop (i1) on ty(i1)
          do i1=1,n
            do i0=1,n
              ierr=ierr+abs(b(i0,i1,i2,i3,i4,i5,i6)-a(i0,i1,i2,i3,i4,i5,i6))
            end do
          end do
        end do
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_7a1t_b_c"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_7a1t_b(tname)

character(*) tname
integer i0,i1,i2,i3,i4,i5,i6
integer,parameter :: n=4
integer a(n,n,n,n,n,n,n),b(n,n,n,n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(*,*,*,*,*,*,i) with tx(i)
!$xmp align b(*,i,*,*,*,*,*) with tx(i)

!$xmp loop (i6) on tx(i6)
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
!$xmp loop (i1) on tx(i1)
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
b(1:n,1:n,1:n,1:n,1:n,1:n,1:n)=a(1:n,1:n,1:n,1:n,1:n,1:n,1:n)

ierr=0
!$xmp loop (i6) on tx(i6)
do i6=1,n
  do i5=1,n
    do i4=1,n
      do i3=1,n
        do i2=1,n
!$xmp loop (i1) on tx(i1)
          do i1=1,n
            do i0=1,n
              ierr=ierr+abs(b(i0,i1,i2,i3,i4,i5,i6)-a(i0,i1,i2,i3,i4,i5,i6))
            end do
          end do
        end do
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_7a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_ata_7a1t_c(tname)

character(*) tname
integer i0,i1,i2,i3,i4,i5,i6
integer,parameter :: n=4
integer a(n,n,n,n,n,n,n),b(n,n,n,n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(cyclic) onto p
!$xmp align a(*,*,*,*,*,*,i) with tx(i)
!$xmp align b(*,i,*,*,*,*,*) with tx(i)

!$xmp loop (i6) on tx(i6)
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
!$xmp loop (i1) on tx(i1)
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
b(1:n,1:n,1:n,1:n,1:n,1:n,1:n)=a(1:n,1:n,1:n,1:n,1:n,1:n,1:n)

ierr=0
!$xmp loop (i6) on tx(i6)
do i6=1,n
  do i5=1,n
    do i4=1,n
      do i3=1,n
        do i2=1,n
!$xmp loop (i1) on tx(i1)
          do i1=1,n
            do i0=1,n
              ierr=ierr+abs(b(i0,i1,i2,i3,i4,i5,i6)-a(i0,i1,i2,i3,i4,i5,i6))
            end do
          end do
        end do
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_ata_7a1t_c"
call chk_int(tname, ierr)

end subroutine
