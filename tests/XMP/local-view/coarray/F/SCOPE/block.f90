!$xmp nodes p(3)
!$xmp nodes q(2)=p(2:3)
integer::a[*],b

me=this_image()
a=333
b=444

!!!!!!!!!!!!!!!!!
!!write(*,*) "me,a,b=",me,a,b
!!!!!!!!!!!!!!!!!

block
  integer, save :: a[*]
!!  !$xmp coarray on q :: a
  if (me==1) then
     sync images(2)
     a=20
     a[2]=a
     sync images(2)
  else if (me==2) then
     a=1234
     sync images(1)
     sync images(1)
     b=a
  endif
end block

nerr=0
if (a/=333) nerr=nerr+1
if (me==2) then
   if (b/=20) nerr=nerr+1
else
   if (b/=444) nerr=nerr+1
endif

call final_msg(nerr)

end


    subroutine final_msg(nerr)
      if (nerr==0) then 
         print '("[",i0,"] OK")', this_image()
      else
         print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
      end if
      return
    end subroutine final_msg
