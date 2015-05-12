program main
  !$xmp nodes p(4)
  !$xmp template t(16)
  !$xmp distribute t(block) onto p 
  integer(8):: a(16),x(16),idx(16)
  !$xmp align a(i) with t(i)
  !$xmp align x(i) with t(i)
  !$xmp align idx(i) with t(i)
  integer:: i
  integer(8):: adash(16),xdash(16),idxdash(16)
  integer:: answer = 0

  do i=1,16
     adash(i)=i
     xdash(i)=0
  end do

  idxdash(1)=1
  idxdash(2)=3
  idxdash(3)=2
  idxdash(4)=6
  idxdash(5)=5
  idxdash(6)=4
  idxdash(7)=10
  idxdash(8)=9
  idxdash(9)=8
  idxdash(10)=7
  idxdash(11)=16
  idxdash(12)=15
  idxdash(13)=14
  idxdash(14)=13
  idxdash(15)=12
  idxdash(16)=11

  do i=1,16
     xdash(i)=adash(idxdash(i))
  end do


  !$xmp loop on t(i)
  do i=1,16
     a(i)=i
  end do

  !$xmp loop on t(i)
  do i=1,16
     x(i)=0
  end do

  !$xmp task on p(1)
  idx(1)=1
  idx(2)=3
  idx(3)=2
  idx(4)=6
  !$xmp end task


  !$xmp task on p(2)
  idx(5)=5
  idx(6)=4
  idx(7)=10
  idx(8)=9
  !$xmp end task

  !$xmp task on p(3)
  idx(9)=8
  idx(10)=7
  idx(11)=16
  idx(12)=15
  !$xmp end task

  !$xmp task on p(4)
  idx(13)=14
  idx(14)=13
  idx(15)=12
  idx(16)=11
  !$xmp end task


  call xmp_gather(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(idx))

  !$xmp loop (i) on t(i)
  do i=1,16
     if(x(i).ne.xdash(i)) then
        answer = -1
     endif
  end do

  !$xmp reduction(+:answer)

  !$xmp task on p(1)
  if ( answer /= 0 ) then
     write(*,*) "ERROR"
     call abort
  endif

  write(*,*) "PASS"
  !$xmp end task

end program main
