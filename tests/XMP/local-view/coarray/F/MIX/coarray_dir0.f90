      PROGRAM zzz

!$xmp nodes p(8)
!$xmp nodes q(2,2)=p(4:7)

     integer p1[*], q1[*], s1[*]
     integer p1ok, q1ok, s1ok

!$xmp coarray on p :: p1
!$xmp coarray on q :: q1

      me=this_image()
      p1=0
      q1=0
      s1=0
      sync all         !! needed before TASK directive
                       !! Should the language specification be changed?
!$xmp task on q(1,:)
      me1=this_image()
      sync all
      p1[me1]=555
      !! p1[1]=555 --> p(1): p1=555
      !! p1[2]=555 --> p(2): p1=555

      q1[me1]=444
      !! q1[1]=444 --> q(1)=p(4): q1=444
      !! q1[2]=444 --> q(2)=p(5): q1=444

      s1[me1]=333
      !! s1[1]=333 --> q(1,1)=p(4): p1=333
      !! s1[2]=333 --> q(1,2)=p(6): p1=333
!$xmp end task
      sync all

!-------------------------- check
      nerr=0

      p1ok=0
      q1ok=0
      s1ok=0

      select case (me)
      case (1)
         p1ok=555
      case (2)
         p1ok=555
      case (3)
      case (4)
         q1ok=444
         s1ok=333
      case (5)
         q1ok=444
      case (6)
         s1ok=333
      case (7)
      end select

      if (p1 /= p1ok) then
         nerr=nerr+1
         write(*,200) me,"p1",p1ok,p1
      endif
      if (q1 /= q1ok) then
         nerr=nerr+1
         write(*,200) me,"q1",q1ok,q1
      endif
      if (s1 /= s1ok) then
         nerr=nerr+1
         write(*,200) me,"s1",s1ok,s1
      endif

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
