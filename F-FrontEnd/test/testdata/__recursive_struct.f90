       module recursive_struct
         type t
           integer :: n
           type(t),pointer :: next
         end type t
       end module recursive_struct
 
