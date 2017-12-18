MODULE mod1

INTERFACE
  SUBROUTINE gcl_AddFieldToHaloExchange_Wrapper() &
  BIND(c, name='communicationwrapper_add_field_to_halo_exchange')
    USE, INTRINSIC :: iso_c_binding
    TYPE(C_PTR), value    :: field

  END SUBROUTINE gcl_AddFieldToHaloExchange_Wrapper
END INTERFACE

END MODULE mod1
