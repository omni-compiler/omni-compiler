      PROGRAM zzz
      include 'xmp_coarray.h'

!$xmp nodes p(3)
!$xmp nodes q(1,2)=p(2:3)

     integer s1[*], a1(10,20)[*], a2(10,20)[2,*], a3(10,20)[3,*]
     integer t1[*], b1(10,20)[*], b2(10,20)[2,*], b3(10,20)[3,*]

!!!! syntax error in F_Front
!$xmp coarray on p :: s1,a1,a2,a3
!$xmp coarray on q :: t1,b1,b2,b3

      s1=0
      a1=0
      a2=0
      a3=0

      me=this_image()

!$xmp task on q(1,:)
      write(*,*) me, this_image()

!$xmp end task

      END
