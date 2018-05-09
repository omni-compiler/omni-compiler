program main
  !$xmp nodes alls(8)
  !$xmp nodes sub(4)=alls(1:4)
  !$xmp nodes sub2(3,2)=alls(2:7)

  parameter(N=100)
  real*8 sum, max, min
  real*8 sum1, max1, min1
  real*8 sumx, maxx, minx
  real*8 data(N)
  real*8,parameter:: eps = 0.00000001

!!---------------------------- init
  nerr=0
  me=this_image()
  do i=1,N
     data(i)=dble(N)*(me-1)+i
  enddo

!!---------------------------- exec #1
  sum1 = 0.0D0
  max1 = -huge(max1)
  min1 = huge(min1)
  do i=1,N
     sum1 = sum1+data(i)
     if (data(i) > max1) max1 = data(i)
     if (data(i) < min1) min1 = data(i)
  enddo

  call co_sum(sum1, sum)
  call co_max(max1, max)
  call co_min(min1, min)
  sumx=320400.d0
  maxx=800.d0
  minx=1.d0

!!---------------------------- check #1
  if (abs(sum-sumx)>eps) then
     nerr=nerr+1
     write(*,200) me,1,"sum",sumx,sum
  endif
  if (abs(max-maxx)>eps) then
     nerr=nerr+1
     write(*,200) me,1,"max",maxx,max
  endif
  if (abs(min-minx)>eps) then
     nerr=nerr+1
     write(*,200) me,1,"min",minx,min
  endif

!!---------------------------- exec #2  nodes p(1:4)
  !$xmp task on sub
  sum1 = 0.0D0
  max1 = -huge(max1)
  min1 = huge(min1)
  do i=1,N
     sum1 = sum1+data(i)
     if (data(i) > max1) max1 = data(i)
     if (data(i) < min1) min1 = data(i)
  enddo

  call co_sum(sum1, sum)
  call co_max(max1, max)
  call co_min(min1, min)
  sumx=80200.d0
  maxx=400.d0
  minx=1.d0
  !$xmp end task

!!---------------------------- check #2
  if (abs(sum-sumx)>eps) then
     nerr=nerr+1
     write(*,200) me,2,"sum",sumx,sum
  endif
  if (abs(max-maxx)>eps) then
     nerr=nerr+1
     write(*,200) me,2,"max",maxx,max
  endif
  if (abs(min-minx)>eps) then
     nerr=nerr+1
     write(*,200) me,2,"min",minx,min
  endif

!!---------------------------- exec #3  nodes p(3)&p(6)
!! nodes sub2(3,2)=alls(2:7)
  !$xmp task on sub2(2,:)
  sum1 = 0.0D0
  max1 = -huge(max1)
  min1 = huge(min1)
  do i=1,N
     sum1 = sum1+data(i)
     if (data(i) > max1) max1 = data(i)
     if (data(i) < min1) min1 = data(i)
  enddo

  call co_sum(sum1, sum)
  call co_max(max1, max)
  call co_min(min1, min)
  sumx=80100.d0
  maxx=600.d0
  minx=201.d0
  !$xmp end task

!!---------------------------- check #3
  if (abs(sum-sumx)>eps) then
     nerr=nerr+1
     write(*,200) me,3,"sum",sumx,sum
  endif
  if (abs(max-maxx)>eps) then
     nerr=nerr+1
     write(*,200) me,3,"max",maxx,max
  endif
  if (abs(min-minx)>eps) then
     nerr=nerr+1
     write(*,200) me,3,"min",minx,min
  endif

!!---------------------------- fin
    if (nerr==0) then
       write(*,100) this_image()
    else
       write(*,110) this_image(), nerr
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)
200 format("[",i0,"] PHASE",i0,": ",a," should be",f10.1," but ",f10.1)

end program
