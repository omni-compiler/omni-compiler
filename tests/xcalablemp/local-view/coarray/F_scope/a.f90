  module xx
    integer,save:: aaa[*]
  end module xx

  subroutine zz(c)
    use xx
    integer,save:: bbb[*]
    integer:: c[*]
    return
  end subroutine zz
