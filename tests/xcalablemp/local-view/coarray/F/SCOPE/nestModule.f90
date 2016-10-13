  module mod1
!!     include "xmp_coarray.h"
    real a[*]
    integer nerr
  end module mod1

  module mod2
    use mod1
!!    character(4) c[*] bug354
    character(4) c(1)[*]
    real al[*] 

  contains
    integer function thisim()
      thisim = this_image()
    end function

    subroutine ff(z)
      real(8) :: z

      al[mod(thisim(),3)+1] = real(z)*3
      a[mod(thisim(),3)+1] = real(z)*3
      sync all

      me=this_image()
      eps=0.00001
      if (me==1 .and. abs(al-90.0)>eps) then
         nerr=nerr+1
         write(*,*) "NG: al[1] should be 90.0 but", al
      endif
      if (me==2 .and. abs(al-30.0)>eps) then
         nerr=nerr+1
         write(*,*) "NG: al[2] should be 30.0 but", al
      endif
      if (me==3 .and. abs(al-60.0)>eps) then
         nerr=nerr+1
         write(*,*) "NG: al[3] should be 60.0 but", al
      endif


    end subroutine ff

  end module mod2

  subroutine sub1(me)
    use mod1
    use mod2
    integer b(10)[*]

    if (me==1) then
       c="one"
    else if (me==2) then
       c="two"
    else
       c="three"
    endif

  end subroutine sub1

  program main1
    use mod2
    double precision d[*]

    nerr=0
    me = thisim()

    !! TEST#1
    call sub1(thisim(),c)
    syncall

    if (c(1)[1].ne."one ") then
       nerr=nerr+1
       write(*,*) "NG[",me,'] c[1] should be "one ".'
    endif
       
    !! TEST#2
    syncall
    if (me==3) then
       d[1] = 10.0d0
       d[2] = d[1]+10.0
       d = 30.0
    endif
    syncall

    call ff(d)

    eps=0.00001
    if (me==1 .and. abs(a-90.0)>eps) then
       nerr=nerr+1
       write(*,*) "NG: a[1] should be 90.0 but", a
    endif
    if (me==2 .and. abs(a-30.0)>eps) then
       nerr=nerr+1
       write(*,*) "NG: a[2] should be 30.0 but", a
    endif
    if (me==3 .and. abs(a-60.0)>eps) then
       nerr=nerr+1
       write(*,*) "NG: a[3] should be 60.0 but", a
    endif

    call final_msg(nerr, me)

  end program main1


  subroutine final_msg(nerr)
!!     include 'xmp_coarray.h'
    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if
    return
  end subroutine final_msg

