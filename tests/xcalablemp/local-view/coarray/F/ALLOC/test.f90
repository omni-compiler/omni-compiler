  real, allocatable :: a(:)[:]
  print *,1
  allocate (a(10)[*])
  print *,2
  allocate (a(10)[*])
  end
