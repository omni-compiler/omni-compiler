      PROGRAM zzz

!$xmp nodes p(8)
!$xmp nodes q(2,2)=p(4:7)

     integer s1[*], a1(10,20)[*], a2(10,20)[2,*], a3(10,20)[3,*]
!!     integer t1[*], b1(10,20)[*], b2(10,20)[2,*], b3(10,20)[3,*]

!$xmp coarray on p :: s1,a1,a2,a3
!!!$xmp coarray on q :: t1,b1,b2,b3

      s1=0
      a1=0
      a2=0
      a3=0

      nerr=0

      me=this_image()
      ni=num_images()

!$xmp task on q(1,:)
      me1=this_image()
      ni1=num_images()

      if (me1==1) then
         if (me/=4) then
            nerr=nerr+1
            write(*,200) me,4,me1
         endif
      else if (me1==2) then
         if (me/=6) then
            nerr=nerr+1
            write(*,200) me,6,me1
         endif
      else
         nerr=nerr+1
         write(*,210) me
      endif

!$xmp end task

200 format("[",i0,"] me1 should be ",i0," but ",i0)
210 format("[",i0,"] illegal run in task on q(1,:)")

    if (nerr==0) then
       write(*,100) me
    else
       write(*,110) me, nerr
       call exit(1)
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

  END PROGRAM zzz
