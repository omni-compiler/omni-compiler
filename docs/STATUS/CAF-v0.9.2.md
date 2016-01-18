                                                                      2015.04.21
                                                                      Ver 0.9.2

              Coarray Fortran features and the current restrictions

1. General
  Major features of Coarray Fortran 1.0 were implemented in XcalableMP/Fortran 
  compiler. The specification is based on the following article:
    John Reid, JKR Associates, UK. Coarrays in the next Fortran Standard.
    ISO/IEC JTC1/SC22/WG5 N1824, April 21, 2010.
  Current restrictions:
    * To use coarray features, built-in header file 'xmp_coarray.h' must be 
      included at the top of the program unit.
    * Both coarray features and XcalableMP directives cannot be used together.
    
2. Declaration
  Either static or allocatable coarray data objects can be used in the program. 
  Use- and host-associations are available but common- or equivalence-
  association are not allowed.
  Current restrictions:
    * Rank (number of dimensions) of an array may not be more than 7.
    * A coarray cannot be of a derived type nor be a structure component.
    * A coarray cannot be of quadruple precision, i.e., 16-byte real or 32-byte 
      complex.
    * Interface block cannot contains any specification of coarrays. To describe
      explicit interface, host-assocication (using internal procedure) can be 
      used instead.
      
2.1  Static Coarray
  E.g.
      real(8) :: a(100,100)[*], s(1000)[2,2,*]
      integer, save :: n[*], m(3)[4,*]
  The data object is allocated previously before the execution of the user 
  program.  A recursive procedure cannot have a non-allocatable coarray without 
  SAVE attribute.
  Current restrictions:
    * Each lower/upper bound of the shape must be such a simple expression that 
      is an integer constant literal, a simple integer constant expression, or a 
      reference of an integer named constant defined with a simple integer 
      constant expression.
    * A coarray cannot be initialized with initialization or with a DATA 
      statement.
    
2.2  Allocatable Coarray
  E.g.
      real(8), allocatable :: a(:,:)[:], s(:)[:]
      integer, allocatable, save :: n[:], m(:)[:,:]
  The data object is allocated with an ALLOCATE statement as follows:
      allocate ( a(100,100)[*], s(1000)[2,2,*] )
  The allocated coarray is deallocated with an explicit DEALLOCATE statement or 
  with an automatic deallocation at the end of the scope of the name unless it 
  has SAVE attribute.
  Current restrictions:
    * A scalar coarray cannot be allocatable.
    * An allocatable coarray as a dummy argument cannot be allocated or 
      deallocated.
    
3. Inter-image communication
  For the performance of communication, it is recommended to use array assignment
  statements and array expressions of coindexed objects as follows:
      a(:) = b(i,:)[k1] * c(:,j)[k2]    !! getting data from images k1 and k2
      if (this_image(1))  d[k3] = e     !! putting data d on k3 from e
  
4. Image control statements
  SYNC ALL and SYNC MEMORY statements are available.
  Current restrictions:
    * SYNC IMAGES, LOCK and UNLOCK, and CRITICAL and END CRITICAL statements are 
      not supported.
    * stat= and errmsg= specifiers of image control statements are not supported.
    * ERROR STOP statement is not supported.
    
5. Incrinsic Functions
  num_images() and this_image() with no arguments are available.
  Current restrictions:
    * image_index, lcobound and ucobound inquiry functions are not supported.
    * this_image with a coarray argument is not supported.
    * atomic_define and atomic_ref subroutines are not supported.

