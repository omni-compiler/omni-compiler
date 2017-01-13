      module post_decl
      contains
        subroutine sub1()
        end subroutine sub1
        subroutine sub2()
        contains
          subroutine sub3()
            LOGICAL :: p
            p = f()
          end subroutine sub3
          logical function f()
            f = .TRUE.
          end function f
        end subroutine sub2
      end module post_decl
