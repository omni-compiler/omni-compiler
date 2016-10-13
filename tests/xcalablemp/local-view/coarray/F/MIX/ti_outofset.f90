      PROGRAM zzz

!$xmp nodes p(*)
!$xmp nodes q(2,2)=p(3:6)

     integer s1[*]
     integer t1[*],t2[2,*]
     integer s1ok, t1ok, t21ok, t22ok

!$xmp coarray on p :: s1
!$xmp coarray on q :: t1,t2

      nerr=0

      me=this_image()
      ni=num_images()

      select case(me)
      case(1); t1ok=0; t21ok=0; t22ok=0
      case(2); t1ok=0; t21ok=0; t22ok=0
      case(3); t1ok=1; t21ok=1; t22ok=1
      case(4); t1ok=2; t21ok=2; t22ok=1
      case(5); t1ok=3; t21ok=1; t22ok=2
      case(6); t1ok=4; t21ok=2; t22ok=2
      case(7); t1ok=0; t21ok=0; t22ok=0
      case(8); t1ok=0; t21ok=0; t22ok=0
      end select

      if (this_image(t1,1)/=t1ok) then
         nerr=nerr+1
         write(*,200) me,1,"this_image(t1,1)",t1ok,this_image(t1,1)
      endif
      if (this_image(t2,1)/=t21ok) then
         nerr=nerr+1
         write(*,200) me,1,"this_image(t2,1)",t21ok,this_image(t2,1)
      endif
      if (this_image(t2,2)/=t22ok) then
         nerr=nerr+1
         write(*,200) me,1,"this_image(t2,2)",t22ok,this_image(t2,2)
      endif

      sync all

!$xmp task on q
      me=this_image()
      ni=num_images()

      select case(me)
      case(1); t1ok=1; t21ok=1; t22ok=1
      case(2); t1ok=2; t21ok=2; t22ok=1
      case(3); t1ok=3; t21ok=1; t22ok=2
      case(4); t1ok=4; t21ok=2; t22ok=2
      case default; t1ok=-999; t21ok=-999; t22ok=-999
      end select

      if (this_image(t1,1)/=t1ok) then
         nerr=nerr+1
         write(*,200) me,2,"this_image(t1,1)",t1ok,this_image(t1,1)
      endif
      if (this_image(t2,1)/=t21ok) then
         nerr=nerr+1
         write(*,200) me,2,"this_image(t2,1)",t21ok,this_image(t2,1)
      endif
      if (this_image(t2,2)/=t22ok) then
         nerr=nerr+1
         write(*,200) me,2,"this_image(t2,2)",t22ok,this_image(t2,2)
      endif

!$xmp end task


!$xmp task on q(2,:)
                                !! on q(2,1)=p(4) and q(2,2)=p(6)
      me=this_image()
      ni=num_images()

      select case(me)
      case(1); t1ok=2; t21ok=2; t22ok=1
      case(2); t1ok=4; t21ok=2; t22ok=2
      case default; t1ok=-999; t21ok=-999; t22ok=-999
      end select

      if (this_image(t1,1)/=t1ok) then
         nerr=nerr+1
         write(*,200) me,3,"this_image(t1,1)",t1ok,this_image(t1,1)
      endif
      if (this_image(t2,1)/=t21ok) then
         nerr=nerr+1
         write(*,200) me,3,"this_image(t2,1)",t21ok,this_image(t2,1)
      endif
      if (this_image(t2,2)/=t22ok) then
         nerr=nerr+1
         write(*,200) me,3,"this_image(t2,2)",t22ok,this_image(t2,2)
      endif

!$xmp end task

200 format("[",i0,"] PHASE",i0,": ",a," should be ",i0," but ",i0)

    if (nerr==0) then
       write(*,100) this_image()
    else
       write(*,110) this_image(), nerr
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

  END PROGRAM zzz
