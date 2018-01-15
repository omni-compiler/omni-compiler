      MODULE m_nest_dtag__derived_type_and_generics_in_module_2
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
        INTERFACE t
          MODULE PROCEDURE f
        END INTERFACE t
        TYPE s
          INTEGER :: v
        END TYPE s
       CONTAINS
        FUNCTION f(a)
          REAL :: f
          REAL :: a
          f = a
        END FUNCTION f
        FUNCTION g(a)
          REAL :: g
          REAL :: a
          g = a
        END FUNCTION g
#endif
      END MODULE m_nest_dtag__derived_type_and_generics_in_module_2

      MODULE m__derived_type_and_generics_in_module_2
        use m_nest_dtag__derived_type_and_generics_in_module_2
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
        TYPE t
          INTEGER :: v
        END TYPE t
        INTERFACE s
          MODULE PROCEDURE g
        END INTERFACE s
        TYPE(t) :: a
        TYPE(s) :: b
#endif
      END MODULE m__derived_type_and_generics_in_module_2

