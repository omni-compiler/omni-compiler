module issue285

  type mytype
    character(len=4) :: tname
    integer :: gpscen
    integer :: cdtyp
  end type mytype
 
  integer, parameter :: n_gpc = 21
 
  type(mytype), parameter :: gpc(n_gpc) =   (/&
    mytype( 'METO' ,  0, 800 ) ,&  
    mytype( 'MET_' ,  0, 900 ) ,&  
    mytype( 'ASI_' , 21, 821 ) ,&  
    mytype( 'GFZ_' , 23, 823 ) ,&  
    mytype( 'GOP_' , 24, 824 ) ,&  
    mytype( 'GOPE' , 24, 924 ) ,&  
    mytype( 'IEEC' , 25, 825 ) ,&  
    mytype( 'LPT_' , 26, 826 ) ,&  
    mytype( 'LPTR' , 26, 926 ) ,&  
    mytype( 'SGN_' , 29, 829 ) ,&  
    mytype( 'SGN1' , 29, 929 ) ,&  
    mytype( 'BKG_' , 30, 830 ) ,&  
    mytype( 'BKGH' , 30, 930 ) ,&  
    mytype( 'ROB_' , 32, 832 ) ,&  
    mytype( 'KNMI' , 33, 833 ) ,&  
    mytype( 'KNM1' , 33, 933 ) ,&  
    mytype( 'NGAA' , 34, 834 ) ,&  
    mytype( 'NGA_' , 34, 934 ) ,&  
    mytype( 'IGE_' , 35, 835 ) ,&  
    mytype( 'ROB_' , 37, 837 ) ,&  
    mytype( 'XXX_' , 99, 899 ) /) 

  integer, parameter :: ngpgfz = gpc(n_gpc)%cdtyp


end module issue285
