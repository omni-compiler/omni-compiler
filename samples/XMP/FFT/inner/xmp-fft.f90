subroutine xmpfft(n, n_nodes, n_threads)
  use common
  implicit none
  integer*8 :: n
  integer :: n_local, n_nodes, n_threads, nx, ny, c_size

  !! get nx & ny
  call pgetnxny(n, nx, ny, n_nodes)

  !$xmp task on p(1)
  write(*,930) "Number of nodes:       ", n_nodes
  write(*,930) "Number of threads/node:", n_threads
  write(*,933) "Vector size, nx, ny:   ", n, nx, ny
930 format(1X,A,I16)
933 format(1X,A,I16,I8,I8)
  !$xmp end task

  !! get n_local & c_size
  n_local = n / n_nodes
  c_size = (max(nx, ny) * 2 + FFTE_NP) * n_threads
  if (n_local >= c_size) then
     c_size = 0
  end if

  call xmpfft1(n, n_local, nx, ny, c_size)
  return
end subroutine xmpfft


subroutine xmpfft1(n, n_local, nx, ny, c_size)
  use common
  implicit none
  real*8, external :: xmp_wtime
  integer, external :: xmp_node_num

  integer*8 :: n
  integer :: i, n_local, nx, ny, c_size

  complex*16 :: a(n_local), b(n_local), w(n_local)
  complex*16 :: c(c_size)

  real*8 maxErr, tmp1, tmp2, tmp3, t0, t1, t2, t3, Gflops

  !!------------------------------------------ generation
  t0 = - xmp_wtime()
  call get_random(a, 2 * n_local, xmp_node_num())
  t0 = t0 + xmp_wtime()

  !!------------------------------------------ forward
  t1 = t1 - xmp_wtime()
  call xmpsettbl(w, nx, ny)
  t1 = t1 + xmp_wtime()

  !$xmp barrier
  t2 = - xmp_wtime()
  call xmpzfft1d(a, b, c, w, n, nx, ny, c_size, .false.)
  !$omp parallel do
  do i = 1, n_local
     a(i) = b(i)
  end do
  t2 = t2 + xmp_wtime()

  !!------------------------------------------ backward
  t3 = - xmp_wtime()
  call xmpsettbl(w, nx, ny)

  call xmpzfft1d(a, b, c, w, n, nx, ny, c_size, .true.)
  !$omp parallel do
  do i = 1, n_local
     a(i) = b(i)
  end do
  t3 = t3 + xmp_wtime()

  !!------------------------------------------ check
  call get_random(b, 2 * n_local, 0)     !! get the same random numbers

  maxErr = 0.0
  !$omp parallel do private(tmp1,tmp2,tmp3),reduction(max:maxErr)
  do i = 1, n_local
     tmp1 = real(a(i)) - real(b(i))
     tmp2 = aimag(a(i)) - aimag(b(i))
     tmp3 = sqrt( tmp1*tmp1 + tmp2*tmp2 )
     if (maxErr < tmp3) maxErr = tmp3
  end do

  !$xmp reduction(max:t0,t1,t2,t3,maxErr)
  if (t2 > 0.0) Gflops = 1e-9 * (5.0 * n * log(real(n)) / log(2.0)) / t2

  !$xmp task on p(1)
  write(*,931) "Generation time:       ", t0
  write(*,931) "Tuning:                ", t1
  write(*,931) "Computing:             ", t2
  write(*,931) "Inverse FFT:           ", t3
  write(*,932) "max(|x-x0|):           ", maxErr
  write(*,931) "Gflop/s:               ", Gflops

931 format(1X,A,F16.3)
932 format(1X,A,ES16.5)
  !$xmp end task

end subroutine xmpfft1


