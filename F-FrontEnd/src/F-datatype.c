/**
 * \file F-ident.c
 */

#include "F-front.h"

TYPE_DESC
new_type_desc()
{
    TYPE_DESC tp;
    tp = XMALLOC(TYPE_DESC,sizeof(*tp));
    return(tp);
}

TYPE_DESC
type_basic(BASIC_DATA_TYPE t)
{
    TYPE_DESC tp;
    assert(t != TYPE_CHAR);

    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = t;
    return tp;
}

TYPE_DESC
type_char(int len)
{
    TYPE_DESC tp;

    /* (len <=0) means character(len=*) in function argument */
    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = TYPE_CHAR;
    assert(len >= 0 || len == CHAR_LEN_UNFIXED);
    TYPE_CHAR_LEN(tp) = len;
    return tp;
}

TYPE_DESC
function_type(TYPE_DESC tp)
{
    TYPE_DESC tq;
    tq = new_type_desc();
    TYPE_BASIC_TYPE(tq) = TYPE_FUNCTION;
    TYPE_REF(tq) = tp;
    return tq;
}

TYPE_DESC
new_type_subr(void)
{
    TYPE_DESC tp;

    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = (BASIC_DATA_TYPE)TYPE_SUBR;

    return tp;
}

TYPE_DESC
struct_type(ID id)
{
    TYPE_DESC tp;
    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = TYPE_STRUCT;
    TYPE_TAGNAME(tp) = id;
    TYPE_TYPE_PARAMS(tp) = NULL;
    TYPE_MEMBER_LIST(tp) = NULL;
    TYPE_REF(tp) = NULL;
    TYPE_IS_DECLARED(tp) = FALSE;
    return tp;
}

TYPE_DESC
wrap_type(TYPE_DESC tp)
{
    TYPE_DESC tq = new_type_desc();
    if (tp == tq) {
        fatal("%s: must be an malloc() problem, "
              "newly alloc'd TYPE_DESC has duplicated address.",
              __func__);
        /* not reached. */
        return NULL;
    }
    TYPE_REF(tq) = tp;
    if (IS_STRUCT_TYPE(tp)) {
        TYPE_BASIC_TYPE(tq) = TYPE_STRUCT;
    } else {
        TYPE_BASIC_TYPE(tq) = TYPE_BASIC_TYPE(tp);
        TYPE_CHAR_LEN(tq) = TYPE_CHAR_LEN(tp);
        TYPE_KIND(tq) = TYPE_KIND(tp);
        if (TYPE_IS_IMPLICIT(tp))
            TYPE_SET_IMPLICIT(tq);
    }

    return tq;
}

void
merge_attributes(TYPE_DESC tp1, TYPE_DESC tp2)
{
    TYPE_ATTR_FLAGS(tp1) |= TYPE_ATTR_FLAGS(tp2);
}

TYPE_DESC
getBaseType(TYPE_DESC tp)
{
    if (TYPE_REF(tp) != NULL) {
        return getBaseType(TYPE_REF(tp));
    } else {
        return tp;
    }
}

BASIC_DATA_TYPE
getBasicType(TYPE_DESC tp)
{
    BASIC_DATA_TYPE typ;
    if (tp == NULL) {
        return TYPE_UNKNOWN;
    }
    typ = TYPE_BASIC_TYPE(tp);
    if (typ == TYPE_UNKNOWN ||
        typ == TYPE_ARRAY) {
        if (TYPE_REF(tp) != NULL) {
            return getBasicType(TYPE_REF(tp));
        } else {
            return typ;
        }
    } else {
        return typ;
    }
}

TYPE_DESC
getBaseStructType(TYPE_DESC td)
{
    if (td == NULL)
        return NULL;
    while ((TYPE_REF(td) && IS_STRUCT_TYPE(TYPE_REF(td))) &&
           (TYPE_TYPE_PARAM_VALUES(td) == NULL)) {
        td = TYPE_REF(td);
    }
    return td;
}

TYPE_DESC array_element_type(TYPE_DESC tp)
{
    if(!IS_ARRAY_TYPE(tp)) fatal("array_element_type: not ARRAY_TYPE");
    while(IS_ARRAY_TYPE(tp)) tp = TYPE_REF(tp);
    return tp;
}

/**
 * check type is omissible, such that
 * no attributes, no memebers, no indexRanage and so on.
 *
 * FIXME:
 * shrink_type() and type_is_omissible() are quick-fix.
 * see shrink_type().
 */
int
type_is_omissible(TYPE_DESC tp, uint32_t attr, uint32_t ext)
{
    // NULL or a terminal type is not omissible.
    if (tp == NULL || TYPE_REF(tp) == NULL)
        return FALSE;
    // The struct type is not omissible.
    if (IS_STRUCT_TYPE(tp))
        return FALSE;
    // The array type is not omissible.
    if (IS_ARRAY_TYPE(tp))
        return FALSE;
    // The function type is not omissible.
    if (IS_FUNCTION_TYPE(tp))
        return FALSE;
    // Co-array is not omissible.
    if (tp->codims != NULL)
        return FALSE;
    // The type has kind, leng, or size is not omissible.
    if (TYPE_KIND(tp) != NULL ||
        TYPE_LENG(tp) != NULL ||
        TYPE_CHAR_LEN(tp) != 0) {
        return FALSE;
    }

#if 0
    // The type has attributes is not omissible.
    if (TYPE_ATTR_FLAGS(tp))
        return FALSE;
    if (TYPE_EXTATTR_FLAGS(tp))
        return FALSE;
#else
    /*
     * not omissible if this type has any attributes that is not
     * included in the given attrribute flags.
     */
    if (TYPE_ATTR_FLAGS(tp) != 0) {
        if ((attr != 0 && (attr & TYPE_ATTR_FLAGS(tp)) == 0) ||
            (attr == 0)) {
            return FALSE;
        }
    }
    if (TYPE_EXTATTR_FLAGS(tp) != 0) {
        if ((ext != 0 && (ext & TYPE_EXTATTR_FLAGS(tp)) == 0) ||
            (ext == 0)) {
            return FALSE;
        }
    }
#endif

    return TRUE;
}

/**
 * shrink TYPE_DESC, ignore the basic_type with a reference and no attributes.
 *
 * FIXME:
 * shrink_type() and type_is_omissible() are quick-fix.
 *
 * These function solve the following problem:
 *  Too long TYPE_REF list created while reading a xmod file,
 *  but F-Frontend expects TYPE_REF list of basic_type shorter than 3.
 *  Thus type attributes of use-associated IDs are discarded.
 *
 * Something wrong with creation of types from xmod file,
 * This quick-fix don't care above, but shrink TYPE_REF list after
 * type is created.
 */
void
shrink_type(TYPE_DESC tp)
{
    TYPE_DESC ref = TYPE_REF(tp);
    while (type_is_omissible(ref, 0, 0)) {
        TYPE_REF(tp) = TYPE_REF(ref);
        ref = TYPE_REF(tp);
    }
}

#if 0
static uint64_t
reduce_type_attr(TYPE_DESC tp) {
    uint64_t f = 0LL;

    if (TYPE_REF(tp) != NULL &&
        TYPE_BASIC_TYPE(TYPE_REF(tp)) != TYPE_STRUCT) {
        /*
         * FIXME:
         *	Do we have to avoid recursive reduction other than the
         *	TYPE_STRUCT ?
         */
        f = reduce_type_attr(TYPE_REF(tp));
    }

    /*
     * upper 32 bit: exflag
     * lower 32 bit: attr
     */
    f |= (((uint64_t)(TYPE_EXTATTR_FLAGS(tp)) << 32)
          & 0xffffffff00000000LL);
    f |= (((uint64_t)TYPE_ATTR_FLAGS(tp)) & 0xffffffffL);

    return f;
}
#endif

static TYPE_DESC
simplify_type_recursively(TYPE_DESC tp, uint32_t attr, uint32_t ext) {
    if (TYPE_REF(tp) != NULL) {
        TYPE_REF(tp) = simplify_type_recursively(TYPE_REF(tp), attr, ext);
    }
    return type_is_omissible(tp, attr, ext) ? TYPE_REF(tp) : tp;
}


/**
 * Reduce redundant type references.
 *
 *	@param	tp	A TYPE_DESC to be reduced.
 *	@return A reduced TYPE_DESC (could be the tp).
 */
TYPE_DESC
reduce_type(TYPE_DESC tp) {
    TYPE_DESC ret = NULL;

    if (tp != NULL) {
#if 0
        uint64_t f = reduce_type_attr(tp);
        uint32_t attr = (uint32_t)(f & 0xffffffffL);
        uint32_t ext = (uint32_t)((f >> 32) & 0xffffffffL);
#else
        uint32_t attr = 0;
        uint32_t ext = 0;
#endif

        ret = simplify_type_recursively(tp, attr, ext);
        if (ret == NULL) {
            fatal("%s: failure.\n", __func__);
            /* not reached. */
            return NULL;
        }
#if 0
        TYPE_ATTR_FLAGS(ret) = attr;
        TYPE_EXTATTR_FLAGS(ret) = ext;
#endif
    }

    return ret;
}


int
char_length(TYPE_DESC tp)
{
    int len = 1;

    while(TYPE_REF(tp)) {
        expv vs = TYPE_DIM_SIZE(tp);
        if(vs && EXPV_CODE(vs) == INT_CONSTANT) {
            if(EXPV_INT_VALUE(vs) < 0)
                return -1;
            len *= EXPV_INT_VALUE(vs);
        } else
            return -1;
        tp = TYPE_REF(tp);
    }
    assert(IS_CHAR(tp));
    return len * TYPE_CHAR_LEN(tp);
}

int
type_is_possible_dreal(TYPE_DESC tp)
{
    expv kind, kind1;

    if(TYPE_REF(tp))
        return type_is_possible_dreal(TYPE_REF(tp));

    if(TYPE_BASIC_TYPE(tp) == TYPE_DREAL)
        return TRUE;

    if(TYPE_BASIC_TYPE(tp) != TYPE_REAL)
        return FALSE;

    kind = TYPE_KIND(tp);
    if(kind == NULL)
        return FALSE;
    kind1 = expv_reduce(kind, TRUE);
    if(kind1 == NULL)
        return FALSE;
    if(expr_is_constant(kind) && EXPV_CODE(kind) == INT_CONSTANT) {
        return (EXPV_INT_VALUE(kind) == KIND_PARAM_DOUBLE);
    }
    /* treat unreducable value as double */
    return TRUE;
}

int
is_array_size_adjustable(TYPE_DESC tp)
{
    assert(IS_ARRAY_TYPE(tp));

    while(IS_ARRAY_TYPE(tp) && TYPE_REF(tp)) {
        if(TYPE_IS_ARRAY_ADJUSTABLE(tp))
            return TRUE;
        tp = TYPE_REF(tp);
    }
    return FALSE;
}

int
is_array_shape_assumed(TYPE_DESC tp)
{
    assert(IS_ARRAY_TYPE(tp));

    while(IS_ARRAY_TYPE(tp) && TYPE_REF(tp)) {
        if(TYPE_IS_ARRAY_ASSUMED_SHAPE(tp))
            return TRUE;
        tp = TYPE_REF(tp);
    }
    return FALSE;
}

int
is_array_size_const(TYPE_DESC tp)
{
    expv upper;
    assert(IS_ARRAY_TYPE(tp));

    while(IS_ARRAY_TYPE(tp) && TYPE_REF(tp)) {
        upper = TYPE_DIM_UPPER(tp);
        if(upper == NULL || expr_is_constant(upper) == FALSE)
            return FALSE;
        tp = TYPE_REF(tp);
    }
    return TRUE;
}

TYPE_DESC
find_struct_decl_head(SYMBOL s, TYPE_DESC head)
{
    TYPE_DESC tp;

    for (tp = head; tp != NULL; tp = TYPE_SLINK(tp)) {
        if(strcmp(ID_NAME(TYPE_TAGNAME(tp)),SYM_NAME(s)) == 0)
            return tp;
    }

    return NULL;
}


ID
find_struct_member(TYPE_DESC struct_td, SYMBOL sym)
{
    ID member = NULL;

    if (!IS_STRUCT_TYPE(struct_td)) {
        return NULL;
    }
    struct_td = getBaseStructType(struct_td);

    if (TYPE_PARENT(struct_td)) {
        ID parent = TYPE_PARENT(struct_td);
        if (ID_SYM(parent) != NULL && (strcmp(ID_NAME(parent), SYM_NAME(sym)) == 0)) {
            member = TYPE_PARENT(struct_td);
        }
        if (member == NULL) {
            member = find_struct_member(ID_TYPE(parent), sym);
        }
    }
    if (member) {
        return member;
    }

    FOREACH_MEMBER(member, struct_td) {
        if (strcmp(ID_NAME(member), SYM_NAME(sym)) == 0) {
            return member;
        }
    }
    return NULL;
}

/* int */
/* is_descendant_coindexed(TYPE_DESC tp){ */

/*   ID id; */

/*   if (!tp) return FALSE; */

/*   if (TYPE_IS_COINDEXED(tp)) return TRUE; */

/*   if (IS_STRUCT_TYPE(tp)){ */

/*     FOREACH_MEMBER(id, tp){ */
/*       if (is_descendant_coindexed(ID_TYPE(id))) return TRUE; */
/*     } */

/*     if (TYPE_REF(tp)) return is_descendant_coindexed(TYPE_REF(tp)); */

/*   } */
/*   else if (IS_ARRAY_TYPE(tp)){ */
/*     return is_descendant_coindexed(bottom_type(tp)); */
/*   } */

/*   return FALSE; */
/* } */


int
has_coarray_component(TYPE_DESC tp){

  ID id;

  if (!tp) return FALSE;

  if (IS_STRUCT_TYPE(tp)){

    FOREACH_MEMBER(id, tp){
      if (TYPE_IS_COINDEXED(ID_TYPE(id))) return TRUE;
      if (IS_STRUCT_TYPE(ID_TYPE(id)) &&
	  !TYPE_IS_ALLOCATABLE(ID_TYPE(id)) && !TYPE_IS_POINTER(ID_TYPE(id))){
	return has_coarray_component(ID_TYPE(id));
      }
    }

    if (TYPE_REF(tp)) return has_coarray_component(TYPE_REF(tp));

  }

  return FALSE;
}

/* check type compatiblity of element types */
int
type_is_compatible(TYPE_DESC tp,TYPE_DESC tq)
{
    if(tp == NULL || tq == NULL ||
       IS_ARRAY_TYPE(tp) || IS_ARRAY_TYPE(tq)) return FALSE;
    if(TYPE_BASIC_TYPE(tp) != TYPE_BASIC_TYPE(tq)) {
      if (TYPE_BASIC_TYPE(tp) == TYPE_GENERIC || TYPE_BASIC_TYPE(tq) == TYPE_GENERIC ||
          TYPE_BASIC_TYPE(tp) == TYPE_GNUMERIC_ALL ||
          TYPE_BASIC_TYPE(tq) == TYPE_GNUMERIC_ALL){
          return TRUE;
      }
      else if(TYPE_BASIC_TYPE(tp) == TYPE_DREAL || TYPE_BASIC_TYPE(tq) == TYPE_DREAL) {
            TYPE_DESC tt;
            tt = (TYPE_BASIC_TYPE(tp) == TYPE_DREAL)?tq:tp;
            if(TYPE_BASIC_TYPE(tt) == TYPE_REAL &&
               EXPR_CODE(TYPE_KIND(tt)) == INT_CONSTANT &&
               EXPV_INT_VALUE(TYPE_KIND(tt)) == KIND_PARAM_DOUBLE)
                return TRUE;
        }
        return FALSE;
    }
    if(TYPE_BASIC_TYPE(tp) == TYPE_CHAR){
        int l1 = TYPE_CHAR_LEN(tp);
        int l2 = TYPE_CHAR_LEN(tq);
        if(l1 > 0 && l2 > 0 && l1 != l2) return FALSE;
    }
    return TRUE;
}


/*
 * check 2 expvs have the same value
 *
 * expects 2 expv appears as type parameter value and are already reduced
 */
static int
type_parameter_expv_equals(expv v1, expv v2, int is_pointer_set)
{
    uint32_t i1, i2;

    if (v1 == NULL || v2 == NULL) {
        return FALSE;
    }

    if (EXPR_CODE(v1) == LEN_SPEC_ASTERISC ||
        EXPR_CODE(v1) == F08_LEN_SPEC_COLON) {
        if (is_pointer_set && EXPR_CODE(v1) == F08_LEN_SPEC_COLON) {
            return TRUE;
        } else if (EXPR_CODE(v1) == EXPR_CODE(v2)) {
            return TRUE;
        }
        return FALSE;

    } else if (EXPR_CODE(v2) == LEN_SPEC_ASTERISC ||
               EXPR_CODE(v2) == F08_LEN_SPEC_COLON) {
        return FALSE;

    } else {
        if (EXPR_CODE(v1) == INT_CONSTANT &&
            EXPR_CODE(v2) == INT_CONSTANT) {
            i1 = EXPV_INT_VALUE(v1);
            i2 = EXPV_INT_VALUE(v2);
            if (i1 == i2) {
                return TRUE;
            } else {
                return FALSE;
            }
        }
        /* CANNOT RECOGNIZE THE VALUE OF EXPV, pass */
        return TRUE;
    }
}


static int
type_parameter_values_is_compatible_for_assignment(TYPE_DESC tp1, TYPE_DESC tp2, int is_pointer_set)
{
    ID type_params1, type_params2;
    ID id1, id2;

    assert(tp1 != NULL && TYPE_BASIC_TYPE(tp1) == TYPE_STRUCT);
    assert(tp2 != NULL && TYPE_BASIC_TYPE(tp2) == TYPE_STRUCT);

    type_params1 = TYPE_TYPE_ACTUAL_PARAMS(tp1);
    type_params2 = TYPE_TYPE_ACTUAL_PARAMS(tp2);

    FOREACH_ID(id1, type_params1) {
        id2 = find_ident_head(ID_SYM(id1), type_params2);
        if (id2 == NULL) {
            return FALSE;
        }

        if (!type_parameter_expv_equals(VAR_INIT_VALUE(id1),
                                        VAR_INIT_VALUE(id2),
                                        is_pointer_set)) {
            return FALSE;
        }
    }

    return TRUE;
}


static int
compare_derived_type_name(TYPE_DESC tp1, TYPE_DESC tp2)
{
    ID name1, name2;
    SYMBOL sym1 = NULL, sym2 = NULL, module1 = NULL, module2 = NULL;
    TYPE_DESC btp1, btp2;


    if (tp1 == tp2) {
        return TRUE;
    }
    if (tp2 == NULL) {
        return FALSE;
    }

    btp1 = getBaseType(tp1);
    btp2 = getBaseType(tp2);

    name1 = TYPE_TAGNAME(btp1);
    name2 = TYPE_TAGNAME(btp2);

    if (name1) {
        if (ID_USEASSOC_INFO(name1)) {
            sym1 = ID_ORIGINAL_NAME(name1) ?: ID_SYM(name1);
            module1 = ID_MODULE_NAME(name1);
        } else {
            sym1 = ID_SYM(name1);
            module1 = NULL;
        }
    }

    if (name2) {
        if (ID_USEASSOC_INFO(name2)) {
            sym2 = ID_ORIGINAL_NAME(name2) ?: ID_SYM(name2);
            module2 = ID_MODULE_NAME(name2);
        } else {
            sym2 = ID_SYM(name2);
            module2 = NULL;
        }
    }

    if (debug_flag) {
        fprintf(debug_fp, "   left is '%s', right is '%s'\n",
                sym1?SYM_NAME(sym1):"(null)",
                sym2?SYM_NAME(sym2):"(null)");
        fprintf(debug_fp, "   left module is '%s', right module is '%s'\n",
                module1?SYM_NAME(module1):"(null)",
                module2?SYM_NAME(module2):"(null)");
    }

    if ((sym1 == sym2) && (module1 == module2)) {
        return TRUE;
    } else {
        return FALSE;
    }
}


/*
 * Check type compatibility of derived-types
 */
int
struct_type_is_compatible_for_assignment(TYPE_DESC tp1, TYPE_DESC tp2, int is_pointer_set)
{
    TYPE_DESC btp2;

    assert(tp1 != NULL && TYPE_BASIC_TYPE(tp1) == TYPE_STRUCT);
    assert(tp2 == NULL || TYPE_BASIC_TYPE(tp2) == TYPE_STRUCT);

    if (debug_flag) {
        fprintf(debug_fp,"\ncomparing derived-type %p and %p\n", tp1, tp2);
    }

    if (tp2 == NULL) {
        if (debug_flag) fprintf(debug_fp,"* right side type is null, return false\n");
        return FALSE;
    }

    if (debug_flag) fprintf(debug_fp,"* compare addresses                   ... ");

    if (tp1 == tp2) {
        if (debug_flag) fprintf(debug_fp," match\n");
        return TRUE;
    }
    if (debug_flag) fprintf(debug_fp," not match\n");

    if (debug_flag) fprintf(debug_fp,"* check if left side type is CLASS(*) ... ");

    if (TYPE_TAGNAME(getBaseType(tp1)) == NULL && TYPE_IS_CLASS(getBaseType(tp1))) {
        /*
         * tp1 is CLASS(*)
         */
        if (debug_flag) fprintf(debug_fp," match\n");
        return TRUE;
    }
    if (debug_flag) fprintf(debug_fp," not match\n");

    btp2 = getBaseStructType(tp2);

    if (debug_flag) fprintf(debug_fp,"* compare type names\n");
    if (!compare_derived_type_name(tp1, tp2)) {
        if (debug_flag) fprintf(debug_fp,"                                      ... not match\n");

        if (TYPE_IS_CLASS(tp1) && TYPE_PARENT(btp2) && is_pointer_set) {
            if (debug_flag) fprintf(debug_fp,"* compare PARENT type\n");
            return struct_type_is_compatible_for_assignment(tp1, TYPE_PARENT_TYPE(btp2), is_pointer_set);
        } else {
            if (debug_flag) fprintf(debug_fp,"seems not compatible\n");
            return FALSE;
        }
    }
    if (debug_flag) fprintf(debug_fp,"                                      ... match\n");

    if (TYPE_TYPE_PARAM_VALUES(tp1) == NULL &&
        TYPE_TYPE_PARAM_VALUES(tp2) == NULL) {
        return TRUE;
    } else {
        if (debug_flag) fprintf(debug_fp,"* compare type parameters");
        if (type_parameter_values_is_compatible_for_assignment(tp1, tp2, is_pointer_set)) {
            if (debug_flag) fprintf(debug_fp," match\n");
            return TRUE;
        }
        if (debug_flag) fprintf(debug_fp," not match\n");
    }

    if (debug_flag) fprintf(debug_fp,"seems not compatible\n");

    return FALSE;
}


/* check type compatiblity of element types */
int
type_is_compatible_for_assignment(TYPE_DESC tp1, TYPE_DESC tp2)
{
    BASIC_DATA_TYPE b1;
    BASIC_DATA_TYPE b2;

    assert(TYPE_BASIC_TYPE(tp1));
    b1 = TYPE_BASIC_TYPE(tp1);
    assert(TYPE_BASIC_TYPE(tp2));
    b2 = TYPE_BASIC_TYPE(tp2);

    if(b2 == TYPE_GNUMERIC_ALL)
        return TRUE;

    switch(b1) {
    case TYPE_ARRAY:
    case TYPE_INT:
    case TYPE_REAL:
    case TYPE_DREAL:
    case TYPE_COMPLEX:
    case TYPE_DCOMPLEX:
    case TYPE_GNUMERIC:
    case TYPE_GNUMERIC_ALL:
        if(b2 != TYPE_LOGICAL ||
            (b1 == TYPE_ARRAY && IS_LOGICAL(array_element_type(tp1))))
            return TRUE;
        break;
    case TYPE_LOGICAL:
        if(b2 == TYPE_LOGICAL || b2 == TYPE_GENERIC)
            return TRUE;
        break;
    case TYPE_CHAR:
        if(b2 == TYPE_CHAR || b2 == TYPE_GENERIC)
            return TRUE;
        break;
    case TYPE_STRUCT:
        if (b2 == TYPE_GENERIC)
            return TRUE;
        else if (b2 == TYPE_STRUCT)
            return struct_type_is_compatible_for_assignment(tp1, tp2, FALSE);
        break;
    case TYPE_GENERIC:
        return TRUE;
    default:
        break;
    }
    return FALSE;
}

int
type_is_specific_than(TYPE_DESC tp, TYPE_DESC tq)
{
    if (tp == NULL || tq == NULL || IS_MODULE(tp) || IS_MODULE(tq))
        fatal("invalid argument in type comparison.");

    if (IS_GENERIC_TYPE(tq))
        return TRUE;

    if (IS_GNUMERIC_ALL(tq))
        return (IS_INT_OR_REAL(tp) || IS_COMPLEX(tp));

    if (IS_GNUMERIC(tq))
        return IS_INT_OR_REAL(tp);

    if (IS_ARRAY_TYPE(tq) && IS_ARRAY_TYPE(tp))
        return type_is_specific_than(array_element_type(tq),
                                     array_element_type(tp));

    return FALSE;
}

TYPE_DESC
get_binary_numeric_intrinsic_operation_type(TYPE_DESC t0, TYPE_DESC t1) {
    /*
     * Based on the type promotion rule regulated by the JIS X 3001-1
     * (ISO/IEC 1539-1 : 2004)
     */
    /*
     * Atuned to:
     *	+, -, *, /, **
     */
    TYPE_DESC ret = NULL;
    BASIC_DATA_TYPE b0 = get_basic_type(t0);
    BASIC_DATA_TYPE b1 = get_basic_type(t1);

    if (b0 == TYPE_UNKNOWN || b1 == TYPE_UNKNOWN) {
        fatal("Invalid basic type.");
        /* not reached. */
        return NULL;
    }

    switch (b0) {

        case TYPE_INT: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_COMPLEX: case TYPE_DCOMPLEX:
                case TYPE_GNUMERIC: case TYPE_GNUMERIC_ALL: {
                    ret = t1;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_REAL: {
            switch (b1) {
                case TYPE_INT: {
                    ret = t0;
                    break;
                }
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_COMPLEX: case TYPE_DCOMPLEX:
                case TYPE_GNUMERIC: case TYPE_GNUMERIC_ALL: {
                    ret = t1;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_DREAL: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: {
                    ret = t0;
                    break;
                }
                case TYPE_DREAL:
                case TYPE_COMPLEX: case TYPE_DCOMPLEX:
                case TYPE_GNUMERIC: case TYPE_GNUMERIC_ALL: {
                    ret = t1;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_COMPLEX: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_GNUMERIC: {
                    ret = t0;
                    break;
                }
                case TYPE_COMPLEX: case TYPE_DCOMPLEX:
                case TYPE_GNUMERIC_ALL: {
                    ret = t1;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_DCOMPLEX: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_GNUMERIC:
                case TYPE_COMPLEX: case TYPE_DCOMPLEX: {
                    ret = t0;
                    break;
                }
                case TYPE_GNUMERIC_ALL: {
                    ret = t1;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_GNUMERIC: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_COMPLEX: case TYPE_DCOMPLEX:
                case TYPE_GNUMERIC: {
                    ret = t0;
                    break;
                }
                case TYPE_GNUMERIC_ALL: {
                    ret = t1;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_GNUMERIC_ALL: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_COMPLEX: case TYPE_DCOMPLEX:
                case TYPE_GNUMERIC: case TYPE_GNUMERIC_ALL: {
                    ret = t0;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_STRUCT: {
            /*
             * FIXME:
             *	it depends on operation, could be a user defined
             *	operator. Anyway return NULL at this moment.
             */
            break;
        }

        default: {
            break;
        }

    }

    return ret;
}

TYPE_DESC
get_binary_comparative_intrinsic_operation_type(TYPE_DESC t0, TYPE_DESC t1) {
    /*
     * Based on the type promotion rule regulated by the JIS X 3001-1
     * (ISO/IEC 1539-1 : 2004)
     */
    /*
     * Attuned to:
     *	.GT., .GE., .LT., .LE., >, >=, <, <=.
     */

    int isValid = FALSE;
    BASIC_DATA_TYPE b0 = get_basic_type(t0);
    BASIC_DATA_TYPE b1 = get_basic_type(t1);

    if (b0 == TYPE_UNKNOWN || b1 == TYPE_UNKNOWN) {
        fatal("Invalid basic type.");
        /* not reached. */
        return NULL;
    }

    switch (b0) {

        case TYPE_INT:
        case TYPE_REAL: case TYPE_DREAL:
        case TYPE_GNUMERIC: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_GNUMERIC: {
                    isValid = TRUE;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_CHAR: {
            if (b1 == TYPE_CHAR) {
                isValid = TRUE;
                break;
            }
            break;
        }

        case TYPE_STRUCT: {
            /*
             * FIXME:
             *	it depends on operation, could be a user defined
             *	operator. Anyway return NULL at this moment.
             */
            break;
        }

        default: {
            break;
        }

    }

    return (isValid = TRUE) ? type_basic(TYPE_LOGICAL) : NULL;
}

TYPE_DESC
get_binary_equal_intrinsic_operation_type(TYPE_DESC t0, TYPE_DESC t1) {
    /*
     * Based on the type promotion rule regulated by the JIS X 3001-1
     * (ISO/IEC 1539-1 : 2004)
     */
    /*
     * Attuned to:
     *	.EQ., .NE., ==, /=.
     */

    int isValid = FALSE;
    BASIC_DATA_TYPE b0 = get_basic_type(t0);
    BASIC_DATA_TYPE b1 = get_basic_type(t1);

    if (b0 == TYPE_UNKNOWN || b1 == TYPE_UNKNOWN) {
        fatal("Invalid basic type.");
        /* not reached. */
        return NULL;
    }

    switch (b0) {

        case TYPE_INT:
        case TYPE_REAL: case TYPE_DREAL:
        case TYPE_COMPLEX: case TYPE_DCOMPLEX:
        case TYPE_GNUMERIC: case TYPE_GNUMERIC_ALL: {
            switch (b1) {
                case TYPE_INT:
                case TYPE_REAL: case TYPE_DREAL:
                case TYPE_COMPLEX: case TYPE_DCOMPLEX:
                case TYPE_GNUMERIC: case TYPE_GNUMERIC_ALL: {
                    isValid = TRUE;
                    break;
                }
                default: {
                    break;
                }
            }
            break;
        }

        case TYPE_CHAR: {
            if (b1 == TYPE_CHAR) {
                isValid = TRUE;
                break;
            }
            break;
        }

        case TYPE_STRUCT: {
            /*
             * FIXME:
             *	it depends on operation, could be a user defined
             *	operator. Anyway return NULL at this moment.
             */
            break;
        }

        default: {
            break;
        }

    }

    return (isValid = TRUE) ? type_basic(TYPE_LOGICAL) : NULL;
}

int
type_is_linked(TYPE_DESC tp, TYPE_DESC tlist)
{
    if(tlist == NULL)
        return FALSE;
    do {
        if (tp == tlist)
            return TRUE;
        tlist = TYPE_LINK(tlist);
    } while (tlist != NULL);
    return FALSE;
}

TYPE_DESC
type_link_add(TYPE_DESC tp, TYPE_DESC tlist, TYPE_DESC ttail)
{
    if(ttail == NULL) 
        return tp;
    TYPE_LINK(ttail) = tp;
    ttail = tp;
    while(ttail != TYPE_LINK(ttail) && TYPE_LINK(ttail) != NULL) {
        if(type_is_linked(TYPE_LINK(ttail), tlist))
            break;
        ttail = TYPE_LINK(ttail);
    }
    TYPE_LINK(ttail) = NULL;
    return ttail;
}

TYPE_DESC
copy_type_partially(TYPE_DESC tp, int doCopyAttr) {
    TYPE_DESC ret = NULL;
    if (tp != NULL) {
        ret = new_type_desc();
        *ret = *tp;
        if (doCopyAttr == FALSE) {
            TYPE_ATTR_FLAGS(ret) = 0;
            TYPE_EXTATTR_FLAGS(ret) = 0;
        }
        if (TYPE_REF(tp) != NULL) {
            TYPE_REF(ret) = copy_type_partially(TYPE_REF(tp), doCopyAttr);
        }
    }
    return ret;
}
