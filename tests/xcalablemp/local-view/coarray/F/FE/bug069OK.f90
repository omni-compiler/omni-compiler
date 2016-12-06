  module ca
    real a(10,10)[*]
  end module ca

  program bug068OK
    use ca
    a(3,2)[1]=a(5,6)
  end program bug068OK
