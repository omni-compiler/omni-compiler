program test
    character(len=10),dimension(5)::a
    a(1) = "1234567890"
    print *,a(1)(3:5)
end

