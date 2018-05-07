subroutine foo
  interface
     subroutine bar(b)
       real b(:)
     end subroutine bar
  end interface
  real a(100)
  call bar(a(:))
end subroutine foo

subroutine x
  INTERFACE
     SUBROUTINE s(sub)
       INTERFACE
          SUBROUTINE sub()
          END SUBROUTINE sub
       END INTERFACE
     END SUBROUTINE s
  END INTERFACE
  
  EXTERNAL r
  CALL s(r)
END subroutine x
