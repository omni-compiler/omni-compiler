      PROGRAM zzz

!$xmp nodes p(8)
!$xmp nodes q(2,2)=p(4:7)

     integer s1[*], a1(10,20)[*]
     integer t1[*], b1(10,20)[*]
     integer s1ok, a1ok(10,20)

!!!! syntax error in F_Front
!$xmp coarray on p :: s1,a1
!$xmp coarray on q :: t1,b1

      me=this_image()
      ni=num_images()

      s1=me
      a1=10+me

!-------------------------- check
!$xmp task on q(1,:)
                               !! p(4)&p(6)
      me1=this_image()
      ni1=num_images()

      t1=me1
      b1=10+me1
     
      s1[me1]=t1           !! s1[1]=1, s1[2]=2
      a1(me1,1)[me1]=b1(3,3)   !! a1(1,1)[1]=11, a1(2,1)[2]=12
      a1(4,17)=b1(2,11)[3-me1]+20  !! a1(4,17)[4]=32, a1(4,17)[5]=31

!$xmp end task

      sync all

!$xmp task on q(:,2)
                                !! p(6)&p(7)
      me1=this_image()
      ni1=num_images()

      t1=me1+100
      b1=10+me1+100
     
      s1[me1+5]=t1             !! s1[6]=101, s1[7]=102
      a1(me1,2)[me1+3]=b1(3,3)   !! a1(1,2)[4]=111, a1(2,2)[5]=112
      a1(3,15)=b1(9,10)[3-me1]+50  !! a1(3,15)[6]=162, a1(3,15)[7]=161

!$xmp end task

      sync all
!-------------------------- check
      nerr=0

      s1ok=me
      a1ok=10+me

      select case (me)
      case (1)
         s1ok=1
         a1ok(1,1)=11
      case (2)
         s1ok=2
         a1ok(2,1)=12
      case (3)
      case (4)
         a1ok(4,17)=32
         a1ok(1,2)=111
      case (5)
         a1ok(4,17)=31
         a1ok(2,2)=112
      case (6)
         s1ok=101
         a1ok(3,15)=162
      case (7)
         s1ok=102
         a1ok(3,15)=161
      end select

      if (s1 /= s1ok) then
         nerr=nerr+1
         write(*,200) me,"s1",s1ok,s1
      endif

      do j=1,20
         do i=1,10
            if (a1(i,j) /= a1ok(i,j)) then
               nerr=nerr+1
               write(*,210) me,"s1",i,j,a1ok(i,j),a1(i,j)
            endif
         enddo
      enddo

200 format("[",i0,"] ",a," should be ",i0," but ",i0)
210 format("[",i0,"] ",a,"(",i0,",",i0,") should be ",i0," but ",i0)

    if (nerr==0) then
       write(*,100) me
    else
       write(*,110) me, nerr
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

  END PROGRAM zzz
