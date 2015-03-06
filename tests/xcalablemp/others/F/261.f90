subroutine sub
  TYPE p
     INTEGER, POINTER :: p(:)
  END TYPE p
  TYPE(p) :: t(10)
  allocate (t(1)%p(3))
end subroutine sub
