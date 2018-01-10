program main
  include 'xmp_lib.h'
  !$xmp nodes p(8)
  integer a, procs, v = 0
  procs = xmp_num_nodes()
  a = xmp_node_num()

  !$xmp bcast (a) from p(2) on p(2:procs-1)

  if(xmp_node_num() == 1) then
     if(a /= 1) then
        v = 1 ! false
     endif
  else if(xmp_node_num() == 8) then
     if(a /= 8) then
        v = 1 ! false
     endif
  else
     if(a /= 2) then
        v = 1 ! false
     endif
  endif

  !$xmp reduction (+:v)
  if(v == 0) then
     if(xmp_node_num() == 1) then
        write(*,*) "PASS"
     endif
  else
     if(xmp_node_num() == 1) then
        write(*,*) "ERROR"
     endif
     call exit(1)
  endif
end program main
