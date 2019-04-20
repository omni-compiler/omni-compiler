      PROGRAM zzz
      implicit none
    
!$xmp nodes p(8)
!$xmp nodes q(2,2)=p(4:7)

      integer a1(10,20)[*]
      integer b1(10,20)[*]
      integer a1ok(10,20)
      integer me,me1,ni,ni1
      integer nerr, i,j

!$xmp coarray on p :: a1
!$xmp coarray on q :: b1

      me=this_image()
      ni=num_images()

      a1=10+me
      b1=0

!-------------------------- check

      sync all         !! needed before TASK directive
                       !! Should the language specification be changed?
!$xmp task on q(1,:)
                               !! p(4)&p(6)
      me1=this_image()
      ni1=num_images()

      b1=10+me1
      sync all
      !! p(4): me1=1, b1(:,:)=11
      !! p(6): me1=2, b1(:,:)=12
     
      a1(me1,1)[me1]=b1(3,3)
      !! p(4): a1(1,1)[1]=11 --> p(1): a1(1,1)=11
      !! p(6): a1(2,1)[2]=12 --> p(2): a1(2,1)=12

      a1(4,17)=b1(2,11)[3-me1]+20
      !! p(4): a1(4,17)=b1(2,11)[2]+20=b1(2,11)@q(2)+20
      !!       =b1(2,11)@p(5)+20=0+20=20
      !! p(6): a1(4,17)=b1(2,11)[1]+20=b1(2,11)@q(1)+20
      !!       =b1(2,11)@p(4)+20=11+20=31

!$xmp end task

      sync all

!$xmp task on q(:,2)
                                !! p(6)&p(7)
      me1=this_image()
      ni1=num_images()

      b1=10+me1+100
      sync all
      !! p(6): me1=1, b1(:,:)=111
      !! p(7): me1=2, b1(:,:)=112

      a1(me1,2)[me1+3]=b1(3,3)
      !! p(6): a1(1,2)[4]=111 --> p(4): a1(1,2)=111
      !! p(7); a1(2,2)[5]=112 --> p(5): a1(2,2)=112

      a1(3,15)=b1(9,10)[3-me1]+50
      !! p(6): a1(3,15)=b1(9,10)[2]+50=b1(9,10)@q(2)+50
      !!       =b1(9,10)@p(5)+50=0+50=50
      !! p(7): a1(3,15)=b1(9,10)[1]+50=b1(9,10)@q(1)+50
      !!       =b1(9,10)@p(4)+50=11+50=61

!$xmp end task

      sync all
!-------------------------- check
      nerr=0

      a1ok=10+me

      select case (me)
      case (1)
         a1ok(1,1)=11
      case (2)
         a1ok(2,1)=12
      case (3)
      case (4)
         a1ok(4,17)=20
         a1ok(1,2)=111
      case (5)
         a1ok(2,2)=112
      case (6)
         a1ok(4,17)=31
         a1ok(3,15)=50
      case (7)
         a1ok(3,15)=61
      end select

      do j=1,20
         do i=1,10
            if (a1(i,j) /= a1ok(i,j)) then
               nerr=nerr+1
               write(*,210) me,"a1",i,j,a1ok(i,j),a1(i,j)
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
