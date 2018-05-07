  interface assignment(=)                                                        
     subroutine btn(n,b)                                                         
       integer, intent(out)::n                                                   
       logical, intent(in)::b(:)                                                 
     end subroutine btn                                                          
  end interface                                                                  
                                                                                 
  integer i                                                                      
  logical l(3)                                                                   
                                                                                 
  l = .true.                                                                     
  i = l                                                                          
                                                                                 
  end                                                                            
                                                                                 
  subroutine btn(n,b)                                                            
    integer, intent(out)::n                                                      
    logical, intent(in)::b(:)                                                    
                                                                                 
    n = 0                                                                        
    do i=size(b),1,-1                                                            
       n = n * 2                                                                 
       if (b(i))  n = n + 1                                                      
    enddo                                                                        
    write (*,*) "n=",n                                                           
                                                                                 
  end subroutine btn                                                             

