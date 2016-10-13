  module comod
!!       include 'xmp_coarray.h'

!$xmp nodes p(3)
!$xmp nodes q(1,2)=p(2:3)

     integer s1[*], a1(10,20)[*], a2(10,20)[2,*], a3(10,20)[3,*]

!$xmp coarray on p :: s1,a1,a2,a3

   end module comod

   integer function user(ss1,aa2)
     use comod
     real ss1(10)
     integer aa2(10,20)[2,*]
!$xmp nodes qq(1,2)=p(1:2)
!$xmp coarray on p :: ss1
!$xmp coarray on qq :: aa2

     if (this_image()==1) then
        a1 = a2[2]+s1[3]+aa2[2]
     endif

     return
   end function user

