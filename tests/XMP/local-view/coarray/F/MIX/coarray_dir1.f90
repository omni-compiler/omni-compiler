      PROGRAM zzz
      implicit none

!$xmp nodes p(8)
!$xmp nodes q(2,2)=p(4:7)

      integer s1[*]
      integer t1[*]
      integer s1ok
      integer me,me1,ni,ni1,nerr

!$xmp coarray on p :: s1
!$xmp coarray on q :: t1

      me=this_image()
      ni=num_images()

      s1=me
      t1=0

!-------------------------- check

      sync all         !! needed before TASK directive
                       !! Should the language specification be changed?
!$xmp task on q(1,:)
                               !! p(4)&p(6)
      me1=this_image()
      ni1=num_images()

      t1=me1
      sync all
      !! p(4): me1=1, t1=1
      !! p(6): me1=2, t1=2
     
      s1[me1]=t1*10
      !! p(4): s1[1]=10 --> p(1): s1=10
      !! p(6): s1[2]=20 --> p(2): s1=20

      s1=s1[t1*3]+100
      !! p(4): s1=s1[3]+100=s1@p(3)+100=103
      !! p(6): s1=s1[6]+100=s1@p(6)+100=106

!$xmp end task

      sync all

!$xmp task on q(:,2)
                                !! p(6)&p(7)
      me1=this_image()
      ni1=num_images()

      t1=me1+100
      sync all
      !! p(6): me1=1, t1=101
      !! p(7): me1=2, t1=102

      s1[me1+6]=t1
      !! p(6): s1[7]=101 --> p(7): s1=101
      !! p(7): s1[8]=102 --> p(8): s1=102

!$xmp end task

      sync all
!-------------------------- check
      nerr=0

      s1ok=me

      select case (me)
      case (1)
         s1ok=10
      case (2)
         s1ok=20
      case (3)
      case (4)
         s1ok=103
      case (5)
      case (6)
         s1ok=106
      case (7)
         s1ok=101
      case (8)
         s1ok=102
      end select

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
