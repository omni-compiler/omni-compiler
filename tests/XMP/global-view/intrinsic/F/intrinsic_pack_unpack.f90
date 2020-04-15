program main
  !$xmp nodes p(2,2)
  !$xmp template tp(10,10)
  !$xmp distribute tp(block,block) onto p
  !$xmp nodes q(4)
  !$xmp template tq(100)
  !$xmp distribute tq(block) onto q
  integer(8):: a(10,10),adash(10,10)
  logical:: mask(10,10)
  integer(8):: v(100),vdash(100)
  !$xmp align a(i,j) with tp(i,j)
  !$xmp align mask(i,j) with tp(i,j)
  !$xmp align v(i) with tq(i)
  integer:: i,j,answer=0
  do i=1,10
     do j=1,10
        vdash((i-1)*10+j)=(j-1)*10+(i-1)*2
     enddo
  enddo

  do i=51,100
     vdash(i)=0
  end do

  do i=1,10
     do j=1,10
        adash(i,j)=0
     enddo
  enddo

  do i=1,10
     do j=1,10
        if(mod(j-1,2).eq.0)then
           adash(i,j)=(i-1)*10+j-1
        end if
     end do
  end do

  !$xmp loop on tp(i,j)
  do i=1,10
     do j=1,10
        a(i,j)=(i-1)*10+(j-1)
     end do
  end do

  !$xmp loop on tq(i)
  do i=1,100
     v(i)=0
  end do

  !$xmp loop on  tp(i,j)
  do i=1,10
     do j=1,10
        if(mod(j-1,2).eq.0) then
           mask(i,j)=.true.
        else
           mask(i,j)=.false.
        endif
     end do
  end do

  call xmp_pack(xmp_desc_of(v),xmp_desc_of(a),xmp_desc_of(mask))

  !$xmp loop on tq(i)
  do i=1,100
     if(v(i).ne.vdash(i)) then
        answer = -1
     endif
  end do

  !$xmp reduction(+:answer)

  !$xmp task on p(1,1)
  if ( answer /= 0 ) then
     write(*,*) "ERROR"
     call exit(1)
  endif

  write(*,*) "PASS"
  !$xmp end task

  !$xmp loop on  tp(i,j)
  do j=1,10
     do i=1,10
        a(i,j)=0
     end do
  end do

  call xmp_unpack(xmp_desc_of(a),xmp_desc_of(v),xmp_desc_of(mask))

  !$xmp loop on tp(i,j)
  do i=1,10
     do j=1,10
        if(a(i,j).ne.adash(i,j)) then
           answer = -1
        endif
     end do
  end do

  !$xmp reduction(+:answer)

  !$xmp task on p(1,1)
  if ( answer /= 0 ) then
     write(*,*) "ERROR"
     call exit(1)
  endif

  write(*,*) "PASS"
  !$xmp end task


end program main
