      subroutine rsl_compute_cells( domain, f )
      implicit none
      integer domain            ! domain descriptor
      integer f                 ! function (typed to keep implicit none quiet)
      
      integer nl, min, maj, min_g, maj_g
      integer retval
      integer dummy
      
      call rsl_nl( domain, nl ) ! get nest level for this domain
      call rsl_init_nextcell(domain) ! initializes rsl_next_cell
      call rsl_c_nextcell( domain, min, maj, min_g, maj_g, retval )
      do while ( retval .eq. 1 )
         dummy = f( nl+1, min, maj, min_g, maj_g )
         call rsl_c_nextcell( domain, min, maj, min_g, maj_g, retval )
      enddo
      return
      end
