  program this_image2

!$xmp nodes n1(2,4)   

!$xmp nodes n2(2,2)=n1(:,3:4)  
!! image indexes are: 5,6,7,8

!$xmp nodes n3(2,2)=n1(2,:)    
!! image indexes are: 2,4,6,8 

    real a[*],b[*]

    a=0.0
    b=0.0
    sync all

!$xmp task on n2
    a =this_image()*0.1
    if (this_image()==1) then
       b[2] = a+1.0             !!  b[6]=1.1
    endif
    sync all

    if (this_image()==3) then
       a[4]=b[2]+10.0            !! a[8]=11.1
    endif
    !$xmp task on n2(1,2)
       b[1] = -33.0*this_image()     !! b[7]=-33.0
    !$xmp end task
    sync all

!$xmp end task

!!! check
    me = this_image()
    nerr = 0
    if (me==5) then
       ok_a = 0.1
       ok_b = 0.0
    else if (me==6) then
       ok_a = 0.2
       ok_b = 1.1
    else if (me==7) then
       ok_a = 0.3
       ok_b = -33.0
    else if (me==8) then
       ok_a = 11.1
       ok_b = 0.0
    else
       ok_a = 0.0
       ok_b = 0.0
    end if

    eps = 0.0001
    if (abs(a-ok_a) > eps) then
       nerr=nerr+1
       write(*,200) me,"a",ok_a,a
    endif
    if (abs(b-ok_b) > eps) then
       nerr=nerr+1
       write(*,200) me,"b",ok_b,b
    endif

200 format("[",i0,"] Variable ",a," should be ",f8.3," but ",f8.3)

    if (nerr==0) then
       write(*,100) me
    else
       write(*,110) me, nerr
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

  end program this_image2

