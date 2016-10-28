      MODULE private_and_public_base_types
        TYPE, PRIVATE :: priv0
           INTEGER :: v
        END TYPE priv0
        TYPE, EXTENDS(priv0) :: pub0
           INTEGER :: u
        END TYPE pub0
      END MODULE private_and_public_base_types
