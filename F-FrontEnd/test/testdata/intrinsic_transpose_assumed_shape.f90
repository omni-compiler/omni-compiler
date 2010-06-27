subroutine fft2df (freq)
	complex(4), allocatable, save :: stemp (:, :)
    complex(4) :: freq(:, :)
	freq = transpose (stemp)
end subroutine
