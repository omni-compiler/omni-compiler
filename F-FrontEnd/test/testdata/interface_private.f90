module mmm
  private
  INTERFACE xxx
     SUBROUTINE yyy
     END SUBROUTINE yyy
  END INTERFACE xxx
end module mmm

subroutine sub
  use mmm
  INTERFACE xxx
     SUBROUTINE yyy
     END SUBROUTINE yyy
  END INTERFACE xxx
end subroutine sub
