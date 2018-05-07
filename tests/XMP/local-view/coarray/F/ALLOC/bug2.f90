  subroutine main
    integer :: aaa(20)[*]
    call sub(aaa(3),2,1)
  end subroutine main

  subroutine sub(v, k1,k2)
    integer v,k1,k2
  end subroutine sub

  call main
  end
