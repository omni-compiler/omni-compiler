
subroutine sub1()
  implicit none
  character(len=*), parameter :: file  = 'test.dat'
  character(len=*), parameter :: file2 = 'test.dat  '//achar(0)
  character(len=30) :: date
  complex :: z
  real(8) :: x = 0.866_8
  real(8) :: y
  external handler_print
  logical :: t = .true.
  logical :: f = .true.


  integer, dimension(13) :: ifstat
  integer, dimension(9) :: igmtime
  integer, dimension(3) :: iidate
  integer, dimension(3) :: i3
  integer, dimension(9) :: i9
  integer, dimension(13) :: i13
  real, dimension(2) :: tarray
  real :: result

  integer :: i, a, b
  real :: r, s(5)
  print *, (sizeof(s)/sizeof(r) == 5) ! sizeof

  if(access(file,' ') == 0) print*, 'Test access'

  x = acosd(x)
  x = asind(x)
  x = atand(x)
  x = atan2d(y,x)

  call alarm (3, handler_print, i)
  call alarm (3, handler_print)

  write(*,*) and(t, f)
  write(*,*) and(a, b)

  call chdir("/tmp")
  call chdir("/tmp", i)
! TODO omni doesn't support sub/func intrinsic
! i = chdir("/tmp")

  call chmod('test.dat', 'u+x', i)
  call chmod('test.dat', 'u+x')
! TODO omni doesn't support sub/func intrinsic
! i = chmod('test.dat', 'u+x')

  print *, complex(i, x)
  print *, complex(x, i)
  print *, complex(i, i)
  print *, complex(x, x)

  x = cosd(x)
  x = cotan(x)
  x = cotand(x)

  call ctime(i, date)
! TODO omni doesn't support sub/func intrinsic
!  date = ctime(i)

  print *, dcmplx(i)
  print *, dcmplx(i, i)
  print *, dcmplx(x)
  print *, dcmplx(x, x)
  print *, dcmplx(z)
  print *, dcmplx(z, x)
  print *, dcmplx(z, i)
  
  x = dreal(z)

  call dtime(tarray, result)
! TODO omni sub/func
!  result = dtime(tarray)
 
  call etime(tarray, result)
! TODO omni sub/func
!  result = etime(tarray)

  call fdate(date)
! TODO omni sub/func
!  date = fdate()

  call fget(date(1:1))
  call fget(date(1:1), i)
! TODO omni sub/func
! i = fget(date(1:1))

  call fget(a, date(1:1))
  call fget(a, date(1:1), i)
! TODO omni sub/func
! i = fget(a, date(1:1))

  call flush(10)

  i = fnum(10)

  call fput('c')
  call fput('c', i)
! TODO omni func/sub
! i = fput('c')

  call fput(10, 'c')
  call fput(10, 'c', i)
! TODO omni func/sub
! i = fput(10, 'c')

  call fstat(10, ifstat)
  call fstat(10, ifstat, i)
! TODO omni func/sub
!  i = fstat(10, ifstat)

  call ftell(19, i)
! TODO omni func/sub
!  i = ftell(10)

  call gerror(date)

  call getarg(1, date)

  call getcwd(date)
  call getcwd(date, i)
! TODO omni func/sub
! i = getcwd(date)

  call getenv("HOME", date)

  i = getgid()

  call getlog(date)

  i = getpid()
  i = getuid()

  call gmtime(10, igmtime)

  call hostnm(date) 
  call hostnm(date, i) 
! TODO omni func/sub
!  i = hostnm(date) 

  i = iargc()

  call idate(iidate)

  i = ierrno()

  i = int2(i)
  i = int2(x)
  i = int2(z)

  i = int8(i)
  i = int8(x)
  i = int8(z)

  i = irand(i)

  write(*,*) isatty(10)

  if(isnan(x)) print*, 'Is NaN'

  call itime(i3)

  call kill(1, 2)
  call kill(1, 2, i)
! TODO omni func/sub
!  i = call kill(1, 2)

  call link('/tmp', '/tmp')
  call link('/tmp', '/tmp', i)
! TODO omni func/sub
! i = link('/tmp', '/tmp')

  i = lnblnk("test  ")

  i = loc(x)

  i = long(i)
  i = long(x)
  i = long(z)

  i = lshift(i, 10)

  call lstat("/tmp", i13) 
  call lstat("/tmp", i13, i) 
! TODO omni func/sub
! i = lstat("/tmp", i13) 

  call ltime(10, i9)

  i = malloc(123)

  i = mclock()
  i = mclock8()

  write(*,*) or(t, f)
  write(*,*) or(a, b)

  call perror("warning: ")

  print *, ran(123)
  print *, rand(123)

  call rename('/tmp', '/tmp')
  call rename('/tmp', '/tmp', i)
! TODO omni func/sub
! i = rename('/tmp', '/tmp')

  i = rshift(i, 10)

  print *, secnds (0.0) 

  call second(x)
! TODO omni func/sub
!  x = second()

  call signal (12, handler_print)
  call signal (10, 1)
! TODO omni func/sub
!  i = signal(10, 1)

  x = sind(x)
  z = sind(z)

  call sleep(5)

  call stat("/tmp", i13) 
  call stat("/tmp", i13, i) 
! TODO omni func/sub
! i = stat("/tmp", i13) 


  call symlnk('/tmp', '/tmp')
  call symlnk('/tmp', '/tmp', i)
! TODO omni func/sub
! i = symlnk('/tmp', '/tmp')

  call system('command')
  call system('command', i)
! TODO omni func/sub
!  i = system('command', i)

  x = tand(x)
  z = tand(z)

  i = time()
  i = time8()

  call ttynam(10, date)
! TODO omni func/sub
!  print*, ttynam(10)

  call umask(777)
  call umask(777, i)
! TODO omni func/sub
! i = umask(777)

  call unlink('/tmp/lnk')
  call unlink('/tmp/lnk', i)
! TODO omni func/sub
!  i = unlink('/tmp/lnk')

  write(*,*) xor(t, f)
  write(*,*) xor(a, b)

  call free(10)
  call abort()
  call exit()
  call exit(i)
end subroutine sub1
