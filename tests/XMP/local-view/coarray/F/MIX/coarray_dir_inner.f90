   module comod

!$xmp nodes p(3)
!$xmp nodes q(1,2)=p(2:3)

     integer s1[*], a1(10,20)[*], a2(10,20)[2,*], a3(10,20)[3,*]
     integer t1[*], b1(10,20)[*], b2(10,20)[2,*], b3(10,20)[3,*]

!!!! syntax error in F_Front
!$xmp coarray on p :: s1,a1,a2,a3
!$xmp coarray on q :: t1,b1,b2,b3

   contains
     integer function inner(s1,a2)
       real s1(10)
       integer a2(10,20)[2,*]
       integer aa2(10,20)[2,*]
       integer aaa2(10,20)[2,*]
!$xmp nodes qq(1,2)=p(1:2)
!$xmp coarray on p :: s1
!$xmp coarray on qq :: a2,aa2

       if (this_image()==1) then
          a1 = a2[2]+s1[2]+aa2[2]+aaa2[2]
          b2[2]=t1[2]
       endif

       return
     end function inner

   end module comod

