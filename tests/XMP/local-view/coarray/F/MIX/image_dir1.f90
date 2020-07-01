  !$xmp nodes p(8)
  !$xmp nodes q(5)=p(3:7)

  integer me, n1, n2, n3[*]
!!  integer n2sum, n2max, n2min

  me = this_image()
  n1 = me+10
  n2 = me+20
  n3 = me+30

  if (3<=me.and.me<=7) then
     !$xmp image(q)
     sync all

!!     !$xmp image(q)
!!     sync all(return_status)   !! not supported yet

     if (me==3) then
        sync images(4)
     else if (me==4) then
        !$xmp image(q)
        sync images(1)      !! initial image 3
     endif
     
!!!!! unstable on GASNet/IBV-conduit
!!     !$xmp image(q)
!!     call co_broadcast(n1,1)
!!
!!     !$xmp image(q)
!!     call co_sum(n2,n2sum)
!!
!!     !$xmp image(q)
!!     call co_max(n2,n2max)
!!
!!     !$xmp image(q)
!!     call co_min(n2,n2min)

!!     !$xmp image(q)   !! error detection test
!!     call hello()

!!     !$xmp image(q)
!!     critical          !! not supported yet
!!       n3[1]=n3[1]+n3
!!     end critical      !! not supported yet

  endif

  sync all

  nerr=0

  if (n2.ne.me+20) then
     nerr = nerr+1
     write(*,200) me, "n2", me+20, n2
  endif
  if (n3.ne.me+30) then
     nerr = nerr+1
     write(*,200) me, "n3", me+30, n3
  endif

!!!!! unstable on GASNet/IBV-conduit
!!     if (n1.ne.13) then
!!        nerr = nerr+1
!!        write(*,200) me, "n1", 13, n1
!!     endif
!!    if (n2sum.ne.(23+24+25+26+27)) then
!!       nerr = nerr+1
!!       write(*,200) me, "n2sum", (23+24+25+26+27), n2sum
!!    endif
!!    if (n2max.ne.27) then
!!       nerr = nerr+1
!!       write(*,200) me, "n2max", 27, n2max
!!    endif
!!    if (n2min.ne.23) then
!!       nerr = nerr+1
!!       write(*,200) me, "n2min", 23, n2min
!!    endif
!!  else
!!     if (n1.ne.me+10) then
!!        nerr = nerr+1
!!        write(*,200) me, "n1", me+10, n1
!!     endif
!!  endif

  if (nerr==0) then
     write(*,100) me
  else
     write(*,110) me, nerr
     call exit(1)
  end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)
200 format("[",i1,"] ",a," should be ",i0," but ",i0)

end program
