  program this_image2

!$xmp nodes n1(2,4)   

!! (1,3),(2,3),(1,4),(2,4)
!$xmp nodes n2(2,2)=n1(:,3:4)  

!! (2,1),(2,2),(2,3),(2,4)
!$xmp nodes n3(2,2)=n1(2,:)    

    real a(10,20)[2,*]
    integer imag_a(2), imag_a1, image_a2

    real, allocatable :: b(:)[:,:,:]
    integer imag_b(3), imag_b1, image_b2, image_b3

    common nerr
    nerr=0

    allocate (b(33)[3,2,*])

!!!!!!!!!!!!!!!!!
!!   phase 0
!!!!!!!!!!!!!!!!!

    me = this_image()
    ni = num_images()

    imag_a = this_image(a)
    imag_a1 = this_image(a,1)
    imag_a2 = this_image(a,2)

    imag_b = this_image(b)
    imag_b1 = this_image(b,1)
    imag_b2 = this_image(b,2)
    imag_b3 = this_image(b,3)

!! check image of a
    i = mod(me-1,2)+1
    j = (me-1)/2+1
    if (imag_a(1).ne.i) then
       nerr=nerr+1
       write(*,110) me,"imag_a(1)",i,imag_a(1)
    endif
    if (imag_a(2).ne.j) then
       nerr=nerr+1
       write(*,110) me,"imag_a(2)",i,imag_a(2)
    endif
    if (imag_a1.ne.i) then
       nerr=nerr+1
       write(*,110) me,"imag_a1",i,imag_a1
    endif
    if (imag_a2.ne.j) then
       nerr=nerr+1
       write(*,110) me,"imag_a2",i,imag_a2
    endif

!! check image of b
    i = mod(me-1,3)+1
    k = (me-1)/2/3+1
    j = (me-1-(k-1)*2*3)/3+1
    if (imag_b(1).ne.i) then
       nerr=nerr+1
       write(*,110) me,"imag_b(1)",i,imag_b(1)
    endif
    if (imag_b(2).ne.j) then
       nerr=nerr+1
       write(*,110) me,"imag_b(2)",i,imag_b(2)
    endif
    if (imag_b(3).ne.k) then
       nerr=nerr+1
       write(*,110) me,"imag_b(3)",i,imag_b(3)
    endif
    if (imag_b1.ne.i) then
       nerr=nerr+1
       write(*,110) me,"imag_b1",i,imag_b1
    endif
    if (imag_b2.ne.j) then
       nerr=nerr+1
       write(*,110) me,"imag_b2",i,imag_b2
    end if
    if (imag_b3.ne.k) then
       nerr=nerr+1
       write(*,110) me,"imag_b3",i,imag_b3
    end if

110 format("[",i0,"] ",a," should be ",i0," but ",i0)

!!!!!!!!!!!!!!!!!
!!   phase 1
!!!!!!!!!!!!!!!!!
!$xmp task on n2
    call two_way(imag_a,imag_b,me,1)
!$xmp end task

!!!!!!!!!!!!!!!!!
!!   phase 2
!!!!!!!!!!!!!!!!!
!$xmp task on n3
    call two_way(imag_a,imag_b,me,2)
!$xmp end task


!!!!!!!!!!!!!!!!!
!!   final phase
!!!!!!!!!!!!!!!!!
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program this_image2


!! n1(2,4)   
!! a[2,*], b[3,2,*]
!! phase 1 
!!   nodes n2(2,2) = n1(:,3:*)
!!   n1  (1,3)   (2,3)   (1,4)   (2,4)
!!   me    5       6       7       8   
!!   a   [1,3]   [2,3]   [1,4]   [2,4]
!!   b  [2,2,1] [3,2,1] [1,1,2] [2,1,2]
!!   d    [0]     [1]     [2]     [3]
!! phase 2 
!!   nodes n3(2,2) = n1(2,1:*)    
!!   n1  (2,1)   (2,2)   (2,3)   (2,4)
!!   me    2       4       6       8
!!   a   [2,1]   [2,2]   [2,3]   [2,4]
!!   b  [2,1,1] [1,2,1] [3,2,1] [2,1,2]
!!   d    [0]     [1]     [2]     [3]
  subroutine two_way(imag_a,imag_b,me,phase)
    integer imag_a(2), imag_b(3), me, phase, nerr
    common nerr

    integer d[0:*]
    integer imag_d
    integer table(8,0:3,2)    !! data,d,phase

    !! phase 1
    table(:,0,1)=(/1,3,5,1,3,2,2,1/)
    table(:,1,1)=(/2,3,6,2,3,3,2,1/)
    table(:,2,1)=(/1,4,7,1,4,1,1,2/)
    table(:,3,1)=(/2,4,8,2,4,2,1,2/)

    !! phase 2
    table(:,0,2)=(/2,1,2,2,1,2,1,1/)
    table(:,1,2)=(/2,2,4,2,2,1,2,1/)
    table(:,2,2)=(/2,3,6,2,3,3,2,1/)
    table(:,3,2)=(/2,4,8,2,4,2,1,2/)

    imag_d=this_image(d,1)

    if (me.ne.table(3,imag_d,phase)) then
       nerr=nerr+1
       write(*,120) me,"me",table(3,imag_d,phase),me,phase
    endif

    if (imag_a(1).ne.table(4,imag_d,phase)) then
       nerr=nerr+1
       write(*,120) me,"imag_a(1)",table(4,imag_d,phase),imag_a(1),phase
    endif
    if (imag_a(2).ne.table(5,imag_d,phase)) then
       nerr=nerr+1
       write(*,120) me,"imag_a(2)",table(5,imag_d,phase),imag_a(2),phase
    endif

    if (imag_b(1).ne.table(6,imag_d,phase)) then
       nerr=nerr+1
       write(*,120) me,"imag_b(1)",table(6,imag_d,phase),imag_b(1),phase
    endif
    if (imag_b(2).ne.table(7,imag_d,phase)) then
       nerr=nerr+1
       write(*,120) me,"imag_b(2)",table(7,imag_d,phase),imag_b(2),phase
    endif
    if (imag_b(3).ne.table(8,imag_d,phase)) then
       nerr=nerr+1
       write(*,120) me,"imag_b(3)",table(8,imag_d,phase),imag_b(3),phase
    endif


120 format("[",i0,"] ",a," should be ",i0," but ",i0," (phase ",i0,")")

  end subroutine two_way

