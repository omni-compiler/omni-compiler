subroutine sub
  real v1(2,3)
  pointer (pv1, v1)
  save pv1
  complex,save:: v2(3)

  write(*,*) v1
  return

entry sub1
  do i=1,6
     v2(i)=(1.0,2.0)*i
  enddo
  pv1 = loc(v2)
  return

end subroutine sub


program main
  call sub1
  call sub
end program main
