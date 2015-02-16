program test_mm

  call test_mm_aaa_b_b_b_r8()
  call test_mm_aaa_bc_bc_bc_r8()
  call test_mm_aaa_b_c_bc_c16()
!  call test_mm_aaa_b_c_bc_c8()
  call test_mm_aaa_b_c_bc_i4()
  call test_mm_aaa_b_c_bc_r4()
  call test_mm_aaa_b_c_bc_r8_ax()
  call test_mm_aaa_b_c_bc_r8()
  call test_mm_aaa_b_c_gb_r8()
  call test_mm_aani_b_c_bc_i4()
  call test_mm_aanj_b_c_bc_i4()
  call test_mm_ania_b_c_bc_i4()
  call test_mm_anini_b_c_bc_i4()
  call test_mm_aninj_b_c_bc_i4()
  call test_mm_anja_b_c_bc_i4()
  call test_mm_anjni_b_c_bc_i4()
  call test_mm_anjnj_b_c_bc_i4()
  call test_mm_niaa_b_c_bc_i4()
  call test_mm_niani_b_c_bc_i4()
  call test_mm_nianj_b_c_bc_i4()
  call test_mm_ninia_b_c_bc_i4()
  call test_mm_ninini_b_c_bc_i4()
  call test_mm_nininj_b_c_bc_i4()
  call test_mm_ninja_b_c_bc_i4()
  call test_mm_ninjni_b_c_bc_i4()
  call test_mm_ninjnj_b_c_bc_i4()
  call test_mm_njaa_b_c_bc_i4()
  call test_mm_njani_b_c_bc_i4()
  call test_mm_njanj_b_c_bc_i4()
  call test_mm_njnia_b_c_bc_i4()
  call test_mm_njnini_b_c_bc_i4()
  call test_mm_njninj_b_c_bc_i4()
  call test_mm_njnja_b_c_bc_i4()
  call test_mm_njnjni_b_c_bc_i4()
  call test_mm_njnjnj_b_c_bc_i4()

end program


subroutine test_mm_aaa_b_b_b_r8()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
real(8) a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(block,block) onto p
!$xmp distribute tz(block,block) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aaa_bc_bc_bc_r8()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
real(8) a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(cyclic(2),cyclic(2)) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aaa_b_c_bc_c16()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
complex*16 a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aaa_b_c_bc_c8()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
complex a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aaa_b_c_bc_i4()

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
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aaa_b_c_bc_r4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
real(4) a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aaa_b_c_bc_r8_ax()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
real(8) a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n2+1,n1+1)
!$xmp template ty(n3+1,n2+1)
!$xmp template tz(n3+1,n1+1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j+1,i+1)
!$xmp align b(i,j) with ty(j+1,i+1)
!$xmp align x(i,j) with tz(j+1,i+1)

!$xmp loop (i,j) on tx(j+1,i+1)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (i,j) on ty(j+1,i+1)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (i,j) on tz(j+1,i+1)
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

!$xmp loop (i,j) on tz(j+1,i+1)
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


subroutine test_mm_aaa_b_c_bc_r8()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
real(8) a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aaa_b_c_gb_r8()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
real(8) a(n1,n2),b(n2,n3),x(n1,n3)
integer m1(2)=(/10,11/),m2(2)=(/11,14/)
!$xmp nodes p(2,2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(gblock(m1),gblock(m2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

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


subroutine test_mm_aani_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(*,j) with tz(j)

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

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_aanj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1,n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,*) with tz(i)

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
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_ania_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1,1:2)
!$xmp template tx(n1,n2)
!$xmp template ty(n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(j)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
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


subroutine test_mm_anini_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1,1:2)
!$xmp template tx(n1,n2)
!$xmp template ty(n3)
!$xmp template tz(n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(j)
!$xmp align x(*,j) with tz(j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_aninj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1,1:2)
!$xmp template tx(n1,n2)
!$xmp template ty(n3)
!$xmp template tz(n1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(j)
!$xmp align x(i,*) with tz(i)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_anja_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1,n2)
!$xmp template ty(n2)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
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


subroutine test_mm_anjni_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1,n2)
!$xmp template ty(n2)
!$xmp template tz(n3)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i)
!$xmp align x(*,j) with tz(j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_anjnj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1,1:2)
!$xmp template tx(n1,n2)
!$xmp template ty(n2)
!$xmp template tz(n1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2)) onto q
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
!$xmp loop (i) on ty(i)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_niaa_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (j) on tx(j)
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


subroutine test_mm_niani_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(*,j) with tz(j)

!$xmp loop (j) on tx(j)
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

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_nianj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n2)
!$xmp template ty(n2,n3)
!$xmp template tz(n1)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,*) with tz(i)

!$xmp loop (j) on tx(j)
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
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_ninia_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n2)
!$xmp template ty(n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (j) on tx(j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
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


subroutine test_mm_ninini_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n2)
!$xmp template ty(n3)
!$xmp template tz(n3)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)
!$xmp align x(*,j) with tz(j)

!$xmp loop (j) on tx(j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
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


subroutine test_mm_nininj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n2)
!$xmp template ty(n3)
!$xmp template tz(n1)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)
!$xmp align x(i,*) with tz(i)

!$xmp loop (j) on tx(j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_ninja_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n2)
!$xmp template ty(n2)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,*) with ty(i)
!$xmp align x(i,j) with tz(i,j)

!$xmp loop (j) on tx(j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
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


subroutine test_mm_ninjni_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n2)
!$xmp template ty(n2)
!$xmp template tz(n3)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,*) with ty(i)
!$xmp align x(*,j) with tz(j)

!$xmp loop (j) on tx(j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
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


subroutine test_mm_ninjnj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n2)
!$xmp template ty(n2)
!$xmp template tz(n1)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,*) with ty(i)
!$xmp align x(i,*) with tz(i)

!$xmp loop (j) on tx(j)
  do j=1,n2
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_njaa_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1)
!$xmp template ty(n2,n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,j) with tz(i,j)

  do j=1,n2
!$xmp loop (i) on tx(i)
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


subroutine test_mm_njani_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1)
!$xmp template ty(n2,n3)
!$xmp template tz(n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(*,j) with tz(j)

  do j=1,n2
!$xmp loop (i) on tx(i)
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

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
  do j=1,n3
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njanj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1)
!$xmp template ty(n2,n3)
!$xmp template tz(n1)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,j) with ty(i,j)
!$xmp align x(i,*) with tz(i)

  do j=1,n2
!$xmp loop (i) on tx(i)
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
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_mm_njnia_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1)
!$xmp template ty(n3)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,j) with ty(j)
!$xmp align x(i,j) with tz(i,j)

  do j=1,n2
!$xmp loop (i) on tx(i)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
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


subroutine test_mm_njnini_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n1)
!$xmp template ty(n3)
!$xmp template tz(n3)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,j) with ty(j)
!$xmp align x(*,j) with tz(j)

  do j=1,n2
!$xmp loop (i) on tx(i)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
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


subroutine test_mm_njninj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n1)
!$xmp template ty(n3)
!$xmp template tz(n1)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,j) with ty(j)
!$xmp align x(i,*) with tz(i)

  do j=1,n2
!$xmp loop (i) on tx(i)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,n3
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_mm_njnja_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n1)
!$xmp template ty(n2)
!$xmp template tz(n1,n3)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic) onto q
!$xmp distribute tz(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,*) with ty(i)
!$xmp align x(i,j) with tz(i,j)

  do j=1,n2
!$xmp loop (i) on tx(i)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
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


subroutine test_mm_njnjni_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n1)
!$xmp template ty(n2)
!$xmp template tz(n3)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,*) with ty(i)
!$xmp align x(*,j) with tz(j)

  do j=1,n2
!$xmp loop (i) on tx(i)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

!$xmp loop (j) on tz(j)
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

!$xmp loop (j) on tz(j)
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


subroutine test_mm_njnjnj_b_c_bc_i4()

integer,parameter :: n1=21, n2=23, n3=25
integer ierr
real(8) err,rn1,rn2,rn3,rn4,rn5,rn6,ra,rb
integer a(n1,n2),b(n2,n3),x(n1,n3)
!$xmp nodes p(*)
!$xmp template tx(n1)
!$xmp template ty(n2)
!$xmp template tz(n1)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp distribute tz(cyclic(2)) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,*) with ty(i)
!$xmp align x(i,*) with tz(i)

  do j=1,n2
!$xmp loop (i) on tx(i)
    do i=1,n1
      a(i,j)=(i-1)*n1+j
    end do
  end do

  do j=1,n3
!$xmp loop (i) on ty(i)
    do i=1,n2
      b(i,j)=(j-1)*n3+i
    end do
  end do

  do j=1,n3
!$xmp loop (i) on tz(i)
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
!$xmp loop (i) on tz(i)
    do i=1,n1
      ra=(i-1)*rn1
      rb=(j-1)*rn3
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2
      ierr=ierr+(x(i,j)-rn6)
    end do
  end do

  call chk_int(ierr)

end subroutine
