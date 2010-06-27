block data blkdata
    integer a(2)
    double precision b(3)
    common /cmn1/a, b
    data a, b /1, 2, 3.0, 4.0, 5.0/
end

block data
    integer b(2)
    common /cmn2/ b
    data b /1, 2/
end
