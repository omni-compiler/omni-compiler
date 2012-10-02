program elsewheretest

    integer :: kcell(1:3,1:4)
    integer :: kmin, kmax
    
    data kcell/ 1, 4, 4, 1, 4, 1, 1, 4, 1, 4, 4, 1/ 

    kmin = 0
    kmax = 3

    where ( kcell(:,:) == 1 )
       kcell(:,:) = kmin
    elsewhere ( kcell(:,:) == 4 )
       kcell(:,:) = kmax
    elsewhere ( kcell(:,:) == 3 )
       kcell(:,:) = 999
    elsewhere
       kcell(:,:) = 23
    end where

end program elsewheretest 


