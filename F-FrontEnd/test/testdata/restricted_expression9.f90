      module foo
          type baz 
              integer m
          end type baz
      contains
          pure function hoge(x)
              integer :: hoge
              type(baz) :: x
              intent(in) :: x
          end function hoge
      end module foo
      subroutine bar(a,n)
          use foo
          ! struct constructor is restricted expression
          character(len=hoge(baz(m = 1))) :: a
      end subroutine bar
