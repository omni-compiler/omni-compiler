MODULE mod1

INTERFACE
  SUBROUTINE gcl_AddFieldToHaloExchange_Wrapper() &
  BIND(c, name='communicationwrapper_add_field_to_halo_exchange')
    USE, INTRINSIC :: iso_c_binding
  END SUBROUTINE gcl_AddFieldToHaloExchange_Wrapper
END INTERFACE

END MODULE mod1
