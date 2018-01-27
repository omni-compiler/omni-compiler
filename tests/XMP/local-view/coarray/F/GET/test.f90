  interface which
     integer function thisisint(n)
       integer n
     end function thisisint
     integer function thisisint8(n)
       integer(8) n
     end function thisisint8
  end interface

  real a(19)
  integer(8) n8

  n = which(3)
  write(*,*) n

  n = which(loc(a))
  write(*,*) n

  n8 = loc(a)
  n = which(n8)
  write(*,*) n

  end

  integer function thisisint(n)
    integer n
    thisisint = 4
  end function thisisint
  
  integer function thisisint8(n)
    integer(8) n
    thisisint8 = 8
  end function thisisint8
