program test_matmul_FJ

  call test_matmul_010()
  call test_matmul_011()
  call test_matmul_012()
  call test_matmul_013()
  call test_matmul_014()
#if ((XMP_MPI_VERSION >= 3) || (XMP_MPI_VERSION == 2 && XMP_MPI_SUBVERSION >= 2))
  call test_matmul_015()
#endif
  call test_matmul_016()
  call test_matmul_017()
  call test_matmul_018()
  call test_matmul_019()
  call test_matmul_020()
#if ((XMP_MPI_VERSION >= 3) || (XMP_MPI_VERSION == 2 && XMP_MPI_SUBVERSION >= 2))
  call test_matmul_021()
#endif
  call test_matmul_022()
  call test_matmul_023()
  call test_matmul_024()
  call test_matmul_025()
  call test_matmul_026()
#if ((XMP_MPI_VERSION >= 3) || (XMP_MPI_VERSION == 2 && XMP_MPI_SUBVERSION >= 2))
  call test_matmul_027()
#endif
  call test_matmul_028()
  call test_matmul_029()
  call test_matmul_030()
  call test_matmul_031()
  call test_matmul_032()
#if ((XMP_MPI_VERSION >= 3) || (XMP_MPI_VERSION == 2 && XMP_MPI_SUBVERSION >= 2))
  call test_matmul_033()
#endif

end program

subroutine test_matmul_010()

  integer,parameter:: M=45
  integer,parameter:: N=54
  integer,parameter:: L=23
  integer*2 a(M,L), b(L,N), x(M,N)
  integer*2 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error

!$xmp nodes pa(2,4)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(8)
!$xmp template ta(M,L)
!$xmp template tb(N,L,L)
!$xmp template tx(N)
!$xmp distribute ta(block,block) onto pa
!$xmp distribute tb(cyclic,cyclic,cyclic) onto pb
!$xmp distribute tx(cyclic(3)) onto px
!$xmp align a(*,j) with ta(*,j)
!$xmp align b(*,j) with tb(j,*,*)
!$xmp align x(*,j) with tx(j)

!$xmp loop (j) on ta(*,j)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (j) on tb(j,*,*)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (j) on tx(j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_011()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*4 a(M,L), b(L,N), x(M,N)
  integer*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/10,13/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(8)
!$xmp template ta(L,L,L)
!$xmp template tb(N,L,L)
!$xmp template tx(N)
!$xmp distribute ta(block,gblock(m1),block) onto pa
!$xmp distribute tb(cyclic,cyclic,cyclic) onto pb
!$xmp distribute tx(cyclic(3)) onto px
!$xmp align a(*,j) with ta(*,j,*)
!$xmp align b(*,j) with tb(j,*,*)
!$xmp align x(*,j) with tx(j)

!$xmp loop (j) on ta(*,j,*)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (j) on tb(j,*,*)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (j) on tx(j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_012()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*8 a(M,L), b(L,N), x(M,N)
  integer*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/10,15/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2) = pa(1:2,1,2)
!$xmp nodes px(4) = pa(:,:,1)
!$xmp template ta(L,L+2,L)
!$xmp template tb(L+3)
!$xmp template tx(N+4)
!$xmp distribute ta(block,gblock(m1),block) onto pa
!$xmp distribute tb(block) onto pb
!$xmp distribute tx(cyclic(3)) onto px
!$xmp align a(*,j) with ta(*,j+1,*)
!$xmp align b(i,*) with tb(i+2)
!$xmp align x(*,j) with tx(j+3)

!$xmp loop (j) on ta(*,j+1,*)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i) on tb(i+2)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (j) on tx(j+3) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_013()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*4 a(M,L), b(L,N), x(M,N)
  real*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/10,13/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(8)
!$xmp nodes px(2,2,2)
!$xmp template ta(L,L,L)
!$xmp template tb(L)
!$xmp template tx(M,L,N)
!$xmp distribute ta(block,gblock(m1),block) onto pa
!$xmp distribute tb(block) onto pb
!$xmp distribute tx(block,block,block) onto px
!$xmp align a(*,j) with ta(*,j,*)
!$xmp align b(i,*) with tb(i)
!$xmp align x(i,j) with tx(i,*,j)
!$xmp shadow a(0,2)
!$xmp shadow x(1,1)

!$xmp loop (j) on ta(*,j,*)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i) on tb(i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,*,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_014()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*8 a(M,L), b(L,N), x(M,N)
  real*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/10,13/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(8)
!$xmp nodes px(2,2,2)
!$xmp template ta(M,L)
!$xmp template tb(L)
!$xmp template tx(M,L,N)
!$xmp distribute ta(block,cyclic) onto pa
!$xmp distribute tb(block) onto pb
!$xmp distribute tx(block,cyclic,block) onto px
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,*) with tb(i)
!$xmp align x(i,j) with tx(i,*,j)

!$xmp loop (i,j) on ta(i,j)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i) on tb(i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,13)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,*,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_015()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  complex*16 a(M,L), b(L,N), x(M,N)
  complex*16 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/10,13/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(2,2,2)
!$xmp template ta(M,L+1)
!$xmp template tb(L,L+2,N)
!$xmp template tx(M+3,L,N)
!$xmp distribute ta(block,cyclic) onto pa
!$xmp distribute tb(block,block,cyclic(7)) onto pb
!$xmp distribute tx(block,block,block) onto px
!$xmp align a(i,j) with ta(i,j+1)
!$xmp align b(i,j) with tb(*,i+1,j)
!$xmp align x(i,j) with tx(i+1,*,j)
!$xmp shadow x(2,3)

!$xmp loop (i,j) on ta(i,j+1)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i,j) on tb(*,i+1,j)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i+1,*,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_016()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*2 a(M,L), b(L,N), x(M,N)
  integer*2 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/23,30/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(2,2)=pb(:,1,:)
!$xmp template ta(M,L)
!$xmp template tb(L,L,N)
!$xmp template tx(M,N)
!$xmp distribute ta(block,cyclic) onto pa
!$xmp distribute tb(block,block,cyclic(7)) onto pb
!$xmp distribute tx(block,gblock(m1)) onto px
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(*,i,j)
!$xmp align x(i,j) with tx(i,j)
!$xmp shadow x(2,3)

!$xmp loop (i,j) on ta(i,j)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i,j) on tb(*,i,j)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_017()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*4 a(M,L), b(L,N), x(M,N)
  integer*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(4)=(/10,13,12,18/)
!$xmp nodes pa(8)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(2,4)
!$xmp template ta(M)
!$xmp template tb(L,L,N)
!$xmp template tx(M,N)
!$xmp distribute ta(cyclic) onto pa
!$xmp distribute tb(block,block,cyclic(7)) onto pb
!$xmp distribute tx(block,gblock(m1)) onto px
!$xmp align a(i,*) with ta(i)
!$xmp align b(i,j) with tb(*,i,j)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i) on ta(i)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i,j) on tb(*,i,j)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_018()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*8 a(M,L), b(L,N), x(M,N)
  integer*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(4)=(/10,13,12,18/)
!$xmp nodes pa(8)
!$xmp nodes pb(2,4)
!$xmp nodes px(2,4)
!$xmp template ta(M)
!$xmp template tb(N,L)
!$xmp template tx(M,N)
!$xmp distribute ta(cyclic) onto pa
!$xmp distribute tb(cyclic,block) onto pb
!$xmp distribute tx(block,gblock(m1)) onto px
!$xmp align a(i,*) with ta(i)
!$xmp align b(i,j) with tb(j,i)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i) on ta(i)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i,j) on tb(j,i)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_019()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*4 a(M,L), b(L,N), x(M,N)
  real*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(4)=(/10,13,12,18/)
!$xmp nodes pa(8)
!$xmp nodes pb(2,4)
!$xmp nodes px(2,2,2)
!$xmp template ta(M)
!$xmp template tb(N,L)
!$xmp template tx(N,M,M)
!$xmp distribute ta(cyclic) onto pa
!$xmp distribute tb(cyclic,block) onto pb
!$xmp distribute tx(cyclic,cyclic,block) onto px
!$xmp align a(i,*) with ta(i)
!$xmp align b(i,j) with tb(j,i)
!$xmp align x(i,j) with tx(j,i,*)

!$xmp loop (i) on ta(i)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i,j) on tb(j,i)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(j,i,*) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_020()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*8 a(M,L), b(L,N), x(M,N)
  real*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(4)=(/10,13,12,18/)
!$xmp nodes pa(4,2)
!$xmp nodes pb(2,4)
!$xmp nodes px(2,2,2)
!$xmp template ta(M,L)
!$xmp template tb(N,L)
!$xmp template tx(N,M,M)
!$xmp distribute ta(cyclic,cyclic(12)) onto pa
!$xmp distribute tb(cyclic,block) onto pb
!$xmp distribute tx(cyclic,cyclic,block) onto px
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(j,i)
!$xmp align x(i,j) with tx(j,i,*)

!$xmp loop (i,j) on ta(i,j)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i,j) on tb(j,i)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(j,i,*) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_021()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  complex*16 a(M,L), b(L,N), x(M,N)
  complex*16 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/21,32/)
!$xmp nodes pa(4,2)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(2,2,2)
!$xmp template ta(M,L)
!$xmp template tb(N,N,L)
!$xmp template tx(N,M,M)
!$xmp distribute ta(cyclic,cyclic(12)) onto pa
!$xmp distribute tb(block,gblock(m1),cyclic) onto pb
!$xmp distribute tx(cyclic,cyclic,block) onto px
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(*,j,i)
!$xmp align x(i,j) with tx(j,i,*)

!$xmp loop (i,j) on ta(i,j)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i,j) on tb(*,j,i)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(j,i,*) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_022()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*2 a(M,L), b(L,N), x(M,N)
  integer*2 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/21,32/)
!$xmp nodes pa(4,2)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(2) = pa(2:3,2)
!$xmp template ta(M,L)
!$xmp template tb(N,N,L)
!$xmp template tx(M)
!$xmp distribute ta(cyclic,cyclic(12)) onto pa
!$xmp distribute tb(block,gblock(m1),cyclic) onto pb
!$xmp distribute tx(cyclic(3)) onto px
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(*,j,i)
!$xmp align x(i,*) with tx(i)

!$xmp loop (i,j) on ta(i,j)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(*,j,i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i) on tx(i) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_023()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*4 a(M,L), b(L,N), x(M,N)
  integer*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/21,32/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(8)
!$xmp template ta(L,L,M)
!$xmp template tb(N,N,L)
!$xmp template tx(M)
!$xmp distribute ta(block,block,cyclic(7)) onto pa
!$xmp distribute tb(block,gblock(m1),cyclic) onto pb
!$xmp distribute tx(cyclic(3)) onto px
!$xmp align a(i,j) with ta(j,*,i)
!$xmp align b(i,j) with tb(*,j,i)
!$xmp align x(i,*) with tx(i)

!$xmp loop (i,j) on ta(j,*,i)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(*,j,i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i) on tx(i) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_024()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*8 a(M,L), b(L,N), x(M,N)
  integer*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/21,32/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2,2)=pa(:,:,2)
!$xmp nodes px(8)
!$xmp template ta(L,L,M)
!$xmp template tb(L,N)
!$xmp template tx(M)
!$xmp distribute ta(block,block,cyclic(7)) onto pa
!$xmp distribute tb(cyclic(5),cyclic) onto pb
!$xmp distribute tx(cyclic(3)) onto px
!$xmp align a(i,j) with ta(j,*,i)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,*) with tx(i)

!$xmp loop (i,j) on ta(j,*,i)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(i,j)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i) on tx(i) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_025()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*4 a(M,L), b(L,N), x(M,N)
  real*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/21,32/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2,2)=pa(1,:,:)
!$xmp nodes px(2,2,2)
!$xmp template ta(L,L,M)
!$xmp template tb(L,N)
!$xmp template tx(M,N,N)
!$xmp distribute ta(block,block,cyclic(7)) onto pa
!$xmp distribute tb(cyclic(5),cyclic) onto pb
!$xmp distribute tx(cyclic(3),cyclic(4),cyclic(5)) onto px
!$xmp align a(i,j) with ta(j,*,i)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,j) with tx(i,j,*)

!$xmp loop (i,j) on ta(j,*,i)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(i,j)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j,*) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_026()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*8 a(M,L), b(L,N), x(M,N)
  real*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/11,12/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(4,2)
!$xmp nodes px(2,2,2)
!$xmp template ta(L,M)
!$xmp template tb(L,N)
!$xmp template tx(M,N,N)
!$xmp distribute ta(gblock(m1),cyclic(7)) onto pa
!$xmp distribute tb(cyclic(5),cyclic) onto pb
!$xmp distribute tx(cyclic(3),cyclic(4),cyclic(5)) onto px
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,j) with tx(i,j,*)

!$xmp loop (i,j) on ta(j,i)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(i,j)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j,*) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_027()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  complex*16 a(M,L), b(L,N), x(M,N)
  complex*16 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/11,12/)
  integer m2(4)=(/10,0,0,13/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(4)=pa(2,:)
!$xmp nodes px(2,2,2)
!$xmp template ta(L,M)
!$xmp template tb(L)
!$xmp template tx(M,N,N)
!$xmp distribute ta(gblock(m1),cyclic(7)) onto pa
!$xmp distribute tb(gblock(m2)) onto pb
!$xmp distribute tx(cyclic(3),cyclic(4),cyclic(5)) onto px
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,*) with tb(i)
!$xmp align x(i,j) with tx(i,j,*)

!$xmp loop (i,j) on ta(j,i)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (i) on tb(i)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j,*) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_028()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*2 a(M,L), b(L,N), x(M,N)
  integer*2 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/11,12/)
  integer m2(2)=(/10,13/)
  integer m3(4)=(/7,10,10,20/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(2)=pa(:,2)
!$xmp nodes px(4,2)
!$xmp template ta(L,M)
!$xmp template tb(L)
!$xmp template tx(M,N)
!$xmp distribute ta(gblock(m1),cyclic(7)) onto pa
!$xmp distribute tb(gblock(m2)) onto pb
!$xmp distribute tx(gblock(m3),block) onto px
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,*) with tb(i)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i,j) on ta(j,i)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i) on tb(i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_029()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*4 a(M,L), b(L,N), x(M,N)
  integer*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/17,30/)
  integer m2(2)=(/10,13/)
  integer m3(4)=(/7,10,10,20/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2)=pa(1,2,:)
!$xmp nodes px(4,2)
!$xmp template ta(M,M,L)
!$xmp template tb(L)
!$xmp template tx(M,N)
!$xmp distribute ta(gblock(m1),cyclic,cyclic) onto pa
!$xmp distribute tb(gblock(m2)) onto pb
!$xmp distribute tx(gblock(m3),block) onto px
!$xmp align a(i,j) with ta(i,j,*)
!$xmp align b(i,*) with tb(i)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i,j) on ta(i,j,*)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i) on tb(i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_030()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  integer*8 a(M,L), b(L,N), x(M,N)
  integer*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/17,30/)
  integer m2(2)=(/40,13/)
  integer m3(4)=(/7,10,10,20/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2,2)=pa(:,1,:)
!$xmp nodes px(4,2)
!$xmp template ta(M,M,L)
!$xmp template tb(N,L)
!$xmp template tx(M,N)
!$xmp distribute ta(gblock(m1),cyclic,cyclic) onto pa
!$xmp distribute tb(gblock(m2),cyclic(3)) onto pb
!$xmp distribute tx(gblock(m3),block) onto px
!$xmp align a(i,j) with ta(i,j,*)
!$xmp align b(i,j) with tb(j,i)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i,j) on ta(i,j,*)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(j,i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_031()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*4 a(M,L), b(L,N), x(M,N)
  real*4 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/17,30/)
  integer m2(2)=(/40,13/)
  integer m3(2)=(/23,30/)
  integer m4(2)=(/27,20/)
!$xmp nodes pa(2,2,2)
!$xmp nodes pb(2,2)=pa(2,:,:)
!$xmp nodes px(2,2,2)
!$xmp template ta(M,M,L)
!$xmp template tb(N,L)
!$xmp template tx(M,N,M)
!$xmp distribute ta(gblock(m1),cyclic,cyclic) onto pa
!$xmp distribute tb(gblock(m2),cyclic(3)) onto pb
!$xmp distribute tx(block,gblock(m3),gblock(m4)) onto px
!$xmp align a(i,j) with ta(i,j,*)
!$xmp align b(i,j) with tb(j,i)
!$xmp align x(i,j) with tx(*,j,i)

!$xmp loop (i,j) on ta(i,j,*)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(j,i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(*,j,i) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_032()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  real*8 a(M,L), b(L,N), x(M,N)
  real*8 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/17,30/)
  integer m2(2)=(/40,13/)
  integer m3(2)=(/23,30/)
  integer m4(2)=(/27,20/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(2,2)=pa(:,2:3)
!$xmp nodes px(2,2,2)
!$xmp template ta(M,L)
!$xmp template tb(N,L)
!$xmp template tx(M,N,M)
!$xmp distribute ta(block,block) onto pa
!$xmp distribute tb(gblock(m2),cyclic(3)) onto pb
!$xmp distribute tx(block,gblock(m3),gblock(m4)) onto px
!$xmp align a(*,j) with ta(*,j)
!$xmp align b(i,j) with tb(j,i)
!$xmp align x(i,j) with tx(*,j,i)

!$xmp loop (j) on ta(*,j)
  do j=1, L
     do i=1, M
        a(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = mod((j-1)*M+i-1,11)+1
     enddo
  enddo

!$xmp loop (i,j) on tb(j,i)
  do j=1, N
     do i=1, L
        b(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = mod((j-1)*M+i-1,7)+1
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(*,j,i) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_033()

  integer,parameter:: M=47
  integer,parameter:: N=53
  integer,parameter:: L=23
  complex*16 a(M,L), b(L,N), x(M,N)
  complex*16 aa(M,L), bb(L,N), xx(M,N)
  integer*4 error
  integer m1(2)=(/17,30/)
  integer m2(2)=(/40,13/)
  integer m3(2)=(/23,30/)
  integer m4(2)=(/27,20/)
!$xmp nodes pa(2,4)
!$xmp nodes pb(2,2,2)
!$xmp nodes px(2,2,2)
!$xmp template ta(M,L)
!$xmp template tb(N,N,N)
!$xmp template tx(M,N,M)
!$xmp distribute ta(block,block) onto pa
!$xmp distribute tb(cyclic,cyclic,cyclic) onto pb
!$xmp distribute tx(block,gblock(m3),gblock(m4)) onto px
!$xmp align a(*,j) with ta(*,j)
!$xmp align b(*,j) with tb(j,*,*)
!$xmp align x(i,j) with tx(*,j,i)

!$xmp loop (j) on ta(*,j)
  do j=1, L
     do i=1, M
        a(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo
  do j=1, L
     do i=1, M
        aa(i,j) = dcmplx(mod((j-1)*M+i-1,11)+1,mod((j-1)*M+i-1,13)+1)
     enddo
  enddo

!$xmp loop (j) on tb(j,*,*)
  do j=1, N
     do i=1, L
        b(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo
  do j=1, N
     do i=1, L
        bb(i,j) = dcmplx(mod((j-1)*M+i-1,7)+1,mod((j-1)*M+i-1,5)+1)
     enddo
  enddo

  xx = matmul(aa, bb)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i,j) on tx(*,j,i) reduction(+: error)
  do j=1, N
     do i=1, M
        if(x(i,j) .ne. xx(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine
