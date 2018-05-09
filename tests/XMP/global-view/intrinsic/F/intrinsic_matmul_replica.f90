program test_mm

  call test_mm_aani_b_c_bc_i4_d()
  call test_mm_aanj_b_c_bc_i4_d()
  call test_mm_ania_b_c_bc_i4_d()
  call test_mm_anini_b_c_bc_i4_d()
  call test_mm_aninj_b_c_bc_i4_d()
  call test_mm_anja_b_c_bc_i4_d()
  call test_mm_anjni_b_c_bc_i4_d()
  call test_mm_anjnj_b_c_bc_i4_d()
  call test_mm_niaa_b_c_bc_i4_d()
  call test_mm_niani_b_c_bc_i4_d()
  call test_mm_nianj_b_c_bc_i4_d()
  call test_mm_ninia_b_c_bc_i4_d()
  call test_mm_ninini_b_c_bc_i4_d()
  call test_mm_nininj_b_c_bc_i4_d()
  call test_mm_ninja_b_c_bc_i4_d()
  call test_mm_ninjni_b_c_bc_i4_d()
  call test_mm_ninjnj_b_c_bc_i4_d()
  call test_mm_njaa_b_c_bc_i4_d()
  call test_mm_njani_b_c_bc_i4_d()
  call test_mm_njanj_b_c_bc_i4_d()
  call test_mm_njnia_b_c_bc_i4_d()
  call test_mm_njnini_b_c_bc_i4_d()
  call test_mm_njninj_b_c_bc_i4_d()
  call test_mm_njnja_b_c_bc_i4_d()
  call test_mm_njnjni_b_c_bc_i4_d()
  call test_mm_njnjnj_b_c_bc_i4_d()

end program


subroutine test_mm_aani_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(*,j) with tz(*,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_aanj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,*) with tz(i,*)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_ania_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_anini_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(*,j) with tz(*,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_aninj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(i,*) with tz(i,*)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_anja_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_anjni_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(*,j) with tz(*,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_anjnj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(i,*) with tz(i,*)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_niaa_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_niani_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(*,j) with tz(*,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_nianj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,*) with tz(i,*)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_ninia_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_ninini_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(*,j) with tz(*,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_nininj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(i,*) with tz(i,*)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_ninja_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_ninjni_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(*,j) with tz(*,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_ninjnj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(i,*) with tz(i,*)

!$xmp loop (j) on tx(*,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njaa_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_njani_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(*,j) with tz(*,j)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njanj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,*) with tz(i,*)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njnia_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(i,j) with tz(i,j)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_njnini_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(*,j) with tz(*,j)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njninj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(*,j) with ty(*,j)
!$xmp align x(i,*) with tz(i,*)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njnja_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(i,j) with tz(i,j)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (i,j) on tz(i,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_njnjni_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(*,j) with tz(*,j)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

!$xmp loop (j) on tz(*,j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njnjnj_b_c_bc_i4_d()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,*) with ty(i,*)
!$xmp align x(i,*) with tz(i,*)

  do j=1,n2
!$xmp loop (i) on tx(i,*)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i,*)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      x(i,j)=0
    end do
  end do

  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  ierr=0

  rn1=n1
  rn2=n2
  rn3=n3
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0
  rn5=rn2*(rn2+1)/2.0

  do j=1,n3
!$xmp loop (i) on tz(i,*)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine
