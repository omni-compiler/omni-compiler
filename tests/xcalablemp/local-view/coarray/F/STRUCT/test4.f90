  interface assignment(=)                                                        
     subroutine btn(n,b)                                                         
       type tt
          logical :: val(3)
       end type tt
       integer, intent(out)::n                                                   
       type(tt), intent(in)::b(:)                                                 
     end subroutine btn                                                          
  end interface                                                                  
                                                                                 
  type tt
     logical :: val(3)
  end type tt
  integer i                                                                      
  type(tt):: l(3)                                                                   
                                                                                 
  l%val(1) = .true.
  l%val(2) = .true.
  l%val(3) = .true.
  i = l
                                                                                 
  end                                                                            
                                                                                 
  subroutine btn(n,y)
    type tt
       logical :: val(3)
    end type tt
    integer, intent(out)::n                                                      
    type(tt), intent(in)::y(:)                                                    
    logical :: b(3)
       
    b = y%val
    n = 0                                                                        
    do i=size(b),1,-1                                                            
       n = n * 2                                                                 
       if (b(i))  n = n + 1                                                      
    enddo                                                                        
    write (*,*) "n=",n                                                           
                                                                                 
  end subroutine btn                                                             

