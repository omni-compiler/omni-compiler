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
    assert(len >= 0 || len == CHAR_LEN_UNFIXED || len == CHAR_LEN_ALLOCATABLE);
    TYPE_CHAR_LEN(tp) = len;
    return tp;
}


/*
 * type of the function which returns tp
 */
TYPE_DESC
function_type(TYPE_DESC tp)
{
    TYPE_DESC ftp;
    ftp = new_type_desc();
    TYPE_BASIC_TYPE(ftp) = TYPE_FUNCTION;
    FUNCTION_TYPE_RETURN_TYPE(ftp) = tp;

    if (tp != NULL) {
        if (TYPE_IS_EXTERNAL(tp)) {
            TYPE_SET_EXTERNAL(ftp);
            TYPE_UNSET_EXTERNAL(tp);
        }

        if (TYPE_IS_USED_EXPLICIT(tp)) {
            TYPE_SET_USED_EXPLICIT(ftp);
            TYPE_UNSET_USED_EXPLICIT(tp);
        }
        if (TYPE_IS_OVERRIDDEN(tp)) {
            TYPE_SET_OVERRIDDEN(ftp);
            TYPE_UNSET_OVERRIDDEN(tp);
        }

        if (TYPE_IS_PUBLIC(tp)) {
            TYPE_SET_PUBLIC(ftp);
            TYPE_UNSET_PUBLIC(tp);
        }
        if (TYPE_IS_PRIVATE(tp)) {
            TYPE_SET_PRIVATE(ftp);
            TYPE_UNSET_PRIVATE(tp);
        }
        if (TYPE_IS_PROTECTED(tp)) {
            TYPE_SET_PROTECTED(ftp);
            TYPE_UNSET_PROTECTED(tp);
        }

        if (TYPE_IS_RECURSIVE(tp)) {
            TYPE_SET_RECURSIVE(ftp);
            TYPE_UNSET_RECURSIVE(tp);
        }
        if (TYPE_IS_PURE(tp)) {
            TYPE_SET_PURE(ftp);
            TYPE_UNSET_PURE(tp);
        }
        if (TYPE_IS_ELEMENTAL(tp)) {
            TYPE_SET_ELEMENTAL(ftp);
            TYPE_UNSET_ELEMENTAL(tp);
        }
        if (TYPE_IS_MODULE(tp)) {
            TYPE_SET_MODULE(ftp);
            TYPE_UNSET_MODULE(tp);
        }
        if (TYPE_IS_IMPURE(tp)) {
            TYPE_SET_IMPURE(ftp);
            TYPE_UNSET_IMPURE(tp);
        }
        TYPE_UNSET_SAVE(tp);

        if (FUNCTION_TYPE_IS_VISIBLE_INTRINSIC(tp)) {
            FUNCTION_TYPE_SET_VISIBLE_INTRINSIC(ftp);
        }
        FUNCTION_TYPE_UNSET_VISIBLE_INTRINSIC(tp);
    }


    return ftp;
}


void
replace_or_assign_type(TYPE_DESC * tp, const TYPE_DESC new_tp)
{
    if (tp == NULL || new_tp == NULL) {
        return;
    }

    if (*tp == NULL) {
        *tp = new_tp;
    } else {
        **tp = *new_tp;
    }
}


TYPE_DESC
intrinsic_function_type(TYPE_DESC return_type)
{
    TYPE_DESC ftp = function_type(return_type);
    TYPE_SET_INTRINSIC(ftp);
    return ftp;
}


TYPE_DESC
intrinsic_subroutine_type()
{
    TYPE_DESC ftp = subroutine_type();
    TYPE_SET_INTRINSIC(ftp);
    return ftp;
}


TYPE_DESC
subroutine_type(void)
{
    TYPE_DESC tp;
    TYPE_DESC tq;

    tq = new_type_desc();
    TYPE_BASIC_TYPE(tq) = TYPE_VOID;

    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = TYPE_SUBR;
    FUNCTION_TYPE_RETURN_TYPE(tp) = tq;
    return tp;
}


TYPE_DESC
generic_procedure_type()
{
    TYPE_DESC tp;
    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = TYPE_GENERIC;
    tp = function_type(tp);
    FUNCTION_TYPE_SET_GENERIC(tp);
    return tp;
}


/*
 * function type
 * - return type is TYPE_GENRERIC
 * - set generic flag
 */
TYPE_DESC
generic_function_type()
{
    TYPE_DESC tp;
    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = TYPE_GENERIC;
    tp = function_type(tp);
    FUNCTION_TYPE_SET_GENERIC(tp);
    return tp;
}


TYPE_DESC
generic_subroutine_type()
{
    TYPE_DESC tp;
    TYPE_DESC tq;

    tq = new_type_desc();
    TYPE_BASIC_TYPE(tq) = TYPE_VOID;
    tq = wrap_type(tq);
    TYPE_BASIC_TYPE(tq) = TYPE_GENERIC;
    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = TYPE_SUBR;
    FUNCTION_TYPE_SET_GENERIC(tp);
    return tp;
}


TYPE_DESC
program_type(void)
{
    TYPE_DESC tp = subroutine_type();
    FUNCTION_TYPE_SET_PROGRAM(tp);
    return tp;
}


TYPE_DESC
type_bound_procedure_type(void)
{
    TYPE_DESC ret; /* dummy return type */
    TYPE_DESC tp;
    ret = new_type_desc();
    TYPE_BASIC_TYPE(ret) = TYPE_GENERIC;
    tp = function_type(ret);
    FUNCTION_TYPE_SET_TYPE_BOUND(tp);
    return tp;
}


TYPE_DESC
procedure_type(const TYPE_DESC ftp)
{
    TYPE_DESC tp;
    if (IS_SUBR(ftp)) {
        tp = subroutine_type();
    } else {
        tp = function_type(NULL);
    }
    TYPE_BASIC_TYPE(tp) = TYPE_BASIC_TYPE(ftp);
    TYPE_REF(tp) = ftp;
    TYPE_SET_PROCEDURE(tp);
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
    return get_basic_type(tp);
}

TYPE_DESC
getBaseParameterizedType(TYPE_DESC td)
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

int
type_is_unlimited_class(TYPE_DESC tp)
{
    return (TYPE_IS_CLASS(tp) && TYPE_REF(tp) == NULL);
}

int
type_is_class_of(TYPE_DESC derived_type, TYPE_DESC class)
{
    TYPE_DESC class_base;
    TYPE_DESC derived_type_base;

    if (!TYPE_IS_CLASS(class)) {
        return FALSE;
    }

    class_base = getBaseType(class);
    derived_type_base = getBaseType(derived_type);

    if (class_base == derived_type_base ||
        TYPE_TAGNAME(class_base) == TYPE_TAGNAME(derived_type_base) ||
        ID_SYM(TYPE_TAGNAME(class_base)) == ID_SYM(TYPE_TAGNAME(class_base))) {
        return TRUE;
    } else {
        return FALSE;
    }
}


/*
 * Retrun TRUE if the type is ABSTRACT and is not polymorphic = CLASS
 */
int
type_is_nopolymorphic_abstract(TYPE_DESC tp)
{
    TYPE_DESC btp;

    if (tp == NULL || !IS_STRUCT_TYPE(tp)) {
        return FALSE;
    }

    if (TYPE_IS_CLASS(tp)) {
        return FALSE;
    }

    btp = get_bottom_ref_type(tp);

    if (TYPE_IS_ABSTRACT(btp)) {
        return TRUE;
    }

    return FALSE;
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
    if (IS_PROCEDURE_TYPE(tp))
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

static TYPE_DESC
simplify_type_recursively(TYPE_DESC tp, uint32_t attr, uint32_t ext) {
    if (TYPE_REF(tp) != NULL) {
        TYPE_REF(tp) = simplify_type_recursively(TYPE_REF(tp), attr, ext);
    }
    if (type_is_omissible(tp, attr, ext) == TRUE) {
        return TYPE_REF(tp);
    } else {
        return tp;
    }
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
        uint32_t attr = 0;
        uint32_t ext = 0;

        ret = simplify_type_recursively(tp, attr, ext);
        if (ret == NULL) {
            fatal("%s: failure.\n", __func__);
            /* not reached. */
            return NULL;
        }
    }

    if (IS_PROCEDURE_TYPE(ret)) {
        ID ip;
        FUNCTION_TYPE_RETURN_TYPE(ret) =
                reduce_type(FUNCTION_TYPE_RETURN_TYPE(ret));
        FOREACH_ID(ip, FUNCTION_TYPE_ARGS(ret)) {
            ID_TYPE(ip) = reduce_type(ID_TYPE(ip));
        }
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

/*
 * The array with '*' in for each upper bounds of ranks is implicit shaped array.
 *
 * ex)
 *    INTEGER, PARAMETER :: array(*, 1:*) = (/(/1,2/), (/3,4/), (/5,6/)/)
 */
int
is_array_implicit_shape(TYPE_DESC tp)
{
    assert(IS_ARRAY_TYPE(tp));

    if (!TYPE_IS_PARAMETER(tp) && !IS_ARRAY_TYPE(tp))
        return FALSE;

    while (IS_ARRAY_TYPE(tp) && TYPE_REF(tp)) {
        if (!TYPE_IS_ARRAY_ASSUMED_SIZE(tp))
            return FALSE;
        tp = TYPE_REF(tp);
    }
    if (tp == NULL || IS_ARRAY_TYPE(tp)) {
        /* tp is corrupted! */
        fatal("invalid array type.");
        return FALSE;
    }

    return TRUE;
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

TYPE_DESC
current_struct()
{
    CTL cp;
    FOR_CTLS_BACKWARD(cp) {
        if (CTL_TYPE(cp) == CTL_STRUCT) {
            return CTL_STRUCT_TYPEDESC(cp);
        }
    }
    return NULL;
}


ID
find_struct_member(TYPE_DESC struct_td, SYMBOL sym)
{
    return find_struct_member_allow_private(struct_td, sym, FALSE);
}

ID
find_struct_member_allow_private(TYPE_DESC struct_td, SYMBOL sym, int allow_private_member)
{
    ID member = NULL;

    if (!IS_STRUCT_TYPE(struct_td)) {
        return NULL;
    }

    struct_td = getBaseParameterizedType(struct_td);

    FOREACH_MEMBER(member, struct_td) {
        if (strcmp(ID_NAME(member), SYM_NAME(sym)) == 0) {

            if (ID_CLASS(member) == CL_TYPE_BOUND_PROC) {
                if (!allow_private_member && TYPE_IS_PRIVATE(member)) {
                    /*
                     * If the struct type is defined in the other module,
                     * check accesssibility of the type bound procedure.
                     *
                     * The PRIVATE type-bound procedure can be accessed by:
                     * - the module in which the PRIVATE type-bound procedure is defined
                     * - the submodule of the module
                     */
                    if (TYPE_TAGNAME(struct_td) != NULL &&
                        ID_USEASSOC_INFO(TYPE_TAGNAME(struct_td)) != NULL &&
                        ID_IS_FROM_PARENT_MOD(TYPE_TAGNAME(struct_td)) == FALSE) {
                        error("'%s' is private type bound procedure",
                              SYM_NAME(sym));
                        return NULL;
                    }
                }
            }

            return member;
        }
    }

    if (TYPE_PARENT(struct_td)) {
        ID parent = TYPE_PARENT(struct_td);
        if (ID_SYM(parent) != NULL && (strcmp(ID_NAME(parent), SYM_NAME(sym)) == 0)) {
            return TYPE_PARENT(struct_td);
        }
        if (member == NULL) {
            return find_struct_member_allow_private(ID_TYPE(parent), sym, allow_private_member);
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

/*
 * Check type compatiblity of types softly
 * ignoring type parameters like KIND, LENGTH
 */
int
type_is_soft_compatible(TYPE_DESC tp, TYPE_DESC tq)
{
    if (tp == NULL || tq == NULL ||
       IS_ARRAY_TYPE(tp) || IS_ARRAY_TYPE(tq)) return FALSE;
    if (TYPE_BASIC_TYPE(tp) != TYPE_BASIC_TYPE(tq)) {
      if (TYPE_BASIC_TYPE(tp) == TYPE_GENERIC ||
          TYPE_BASIC_TYPE(tq) == TYPE_GENERIC ||
          TYPE_BASIC_TYPE(tp) == TYPE_GNUMERIC_ALL ||
          TYPE_BASIC_TYPE(tq) == TYPE_GNUMERIC_ALL){
          return TRUE;
      }
      else if(TYPE_BASIC_TYPE(tp) == TYPE_DREAL ||
              TYPE_BASIC_TYPE(tq) == TYPE_DREAL) {
            TYPE_DESC tt;
            tt = (TYPE_BASIC_TYPE(tp) == TYPE_DREAL)?tq:tp;
            if(TYPE_BASIC_TYPE(tt) == TYPE_REAL &&
               EXPR_CODE(TYPE_KIND(tt)) == INT_CONSTANT &&
               EXPV_INT_VALUE(TYPE_KIND(tt)) == KIND_PARAM_DOUBLE)
                return TRUE;
        }
        return FALSE;
    }
    return TRUE;
}

static int
type_is_double(TYPE_DESC tp)
{
    return TYPE_BASIC_TYPE(tp) == TYPE_DREAL ||
            (TYPE_BASIC_TYPE(tp) == TYPE_REAL &&
             EXPR_CODE(TYPE_KIND(tp)) == INT_CONSTANT &&
             EXPV_INT_VALUE(TYPE_KIND(tp)) == KIND_PARAM_DOUBLE);
}


static int
type_parameter_expv_equals(expv v1, expv v2, int is_strict, int for_argument, int for_assignment, int is_pointer_assignment)
{
    uint32_t i1, i2;

    if (v1 == NULL && v2 == NULL) {
        return TRUE;
    }

    if (is_strict == TRUE) {
        /*
         * v1 and v2 should be the same expression
         */

        if (v1 == NULL || v2 == NULL) {
            return FALSE;
        }

        if (EXPR_CODE(v1) != INT_CONSTANT) {
            if (EXPR_CODE(v1) == F08_LEN_SPEC_COLON ||
                EXPR_CODE(v1) == LEN_SPEC_ASTERISC ||
                EXPR_CODE(v1) == F_ASTERISK) {
                if (EXPR_CODE(v1) != EXPR_CODE(v2)) {
                    return FALSE;
                }
            }
        } else {
            if (EXPR_CODE(v2) == F08_LEN_SPEC_COLON ||
                EXPR_CODE(v2) == LEN_SPEC_ASTERISC ||
                EXPR_CODE(v2) == F_ASTERISK) {
                return FALSE;
            }
        }
    }


    if (for_argument == TRUE) {
        if (v1 != NULL && (EXPR_CODE(v1) == LEN_SPEC_ASTERISC || EXPR_CODE(v1) == F_ASTERISK)) {
            return TRUE;
        }
    }

    if (for_assignment) {
#if 0 // to be solved
        if (EXPR_CODE(v1) == LEN_SPEC_ASTERISC ||
            EXPR_CODE(v1) == F08_LEN_SPEC_COLON) {
            if (is_pointer_assignment && EXPR_CODE(v1) == F08_LEN_SPEC_COLON) {
                return TRUE;
            }
        }
#endif
    }

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


/*
 * check 2 expvs have the same value
 *
 * expects 2 expv appears as type parameter value and are already reduced
 */
static int
type_parameter_expv_equals_for_assignment(expv v1, expv v2, int is_pointer_set)
{
    return type_parameter_expv_equals(v1, v2, FALSE, FALSE, TRUE, is_pointer_set);
}


static int
type_parameter_expv_equals_for_argument(expv v1, expv v2)
{
    return type_parameter_expv_equals(v1, v2, FALSE, TRUE, FALSE, FALSE);
}


/*
 * Compare derived-type by their tagnames
 */
int
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
 * Check `parent` and `child` is the same derived-type or
 * `parent` is the parent type of `child`
 */
int
type_is_parent_type(TYPE_DESC parent, TYPE_DESC child)
{
    if (parent == NULL || child == NULL) {
        return FALSE;
    }

    parent = getBaseParameterizedType(parent);
    child = getBaseParameterizedType(child);

    if (compare_derived_type_name(parent, child)) {
        return TRUE;
    } else if (TYPE_PARENT(child)) {
        return type_is_parent_type(parent, TYPE_PARENT_TYPE(child));
    }

    return FALSE;
}


static int
derived_type_parameter_values_is_compatible_for_assignment(TYPE_DESC tp1, TYPE_DESC tp2, int is_pointer_set)
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

        if (!type_parameter_expv_equals_for_assignment(VAR_INIT_VALUE(id1),
                                        VAR_INIT_VALUE(id2),
                                        is_pointer_set)) {
            return FALSE;
        }
    }

    return TRUE;
}


static int
derived_type_parameter_values_is_compatible(TYPE_DESC tp1, TYPE_DESC tp2, int for_argunemt)
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

        if (EXPR_CODE(VAR_INIT_VALUE(id1)) == LEN_SPEC_ASTERISC) {
            continue;
        }

        if (!type_parameter_expv_equals_for_argument(VAR_INIT_VALUE(id1),
                                        VAR_INIT_VALUE(id2))) {
            return FALSE;
        }
    }

    return TRUE;
}


/*
 * Check compatiblity of types with type parameters
 */
int
derived_type_is_compatible(TYPE_DESC left, TYPE_DESC right, int for_argunemt)
{
    if (!compare_derived_type_name(left, right)) {
        if (TYPE_IS_CLASS(left) && TYPE_PARENT(right)) {
            /*
             * compare the parent
             */
            return derived_type_is_compatible(left, TYPE_PARENT_TYPE(right), for_argunemt);
        }
    } else {
        if (TYPE_TYPE_PARAM_VALUES(left) == NULL &&
            TYPE_TYPE_PARAM_VALUES(right) == NULL) {
            return TRUE;
        } else {
            if (derived_type_parameter_values_is_compatible(left, right, for_argunemt)) {
                if (debug_flag) { fprintf(debug_fp," compatible\n"); }
                return TRUE;
            }
            if (debug_flag) { fprintf(debug_fp," not compatible\n"); }
        }
    }
    return FALSE;
}


/*
 * Check types are TKR(type, kind, rank) compatible
 *
 * If for_argunemt is TRUE, `left` is the type of dummy argument, and `right` is the type of arcutal argument
 */
int
type_is_compatible(TYPE_DESC left, TYPE_DESC right,
                   int is_strict,
                   int for_argunemt,
                   int for_assignment,
                   int is_pointer_assignment,
                   int compare_rank,
                   int issue_error)
{
    TYPE_DESC left_basic, right_basic;

    if (left == NULL || right == NULL) {
        if (debug_flag) {
            fprintf(debug_fp, "# unexpected comparison\n");
        }
        return FALSE;
    }

    if (!compare_rank) {
        left = bottom_type(left);
        right = bottom_type(right);
    }

    left_basic = getBaseParameterizedType(left);
    right_basic = getBaseParameterizedType(right);

    /* type_comparison: */

    if (debug_flag) {
        fprintf(debug_fp, "# comparing basic types\n");
    }

    if (IS_ANY_CLASS(left) || IS_ANY_CLASS(right)) {
        fprintf(debug_fp, "# CLASS(*) \n");
        goto rank_compatibility;
    }

    if (TYPE_BASIC_TYPE(left_basic) == TYPE_BASIC_TYPE(right_basic)) {
        goto kind_compatibility;
    }

    if (IS_STRUCT_TYPE(left_basic) && IS_STRUCT_TYPE(right_basic)) {
        if (derived_type_is_compatible(left_basic, right_basic, for_argunemt)) {
            /*
             * derived_type_is_compatible include kind_compatibility
             */
            goto rank_compatibility;
        }
    }

    if (TYPE_BASIC_TYPE(left_basic) == TYPE_GENERIC ||
        TYPE_BASIC_TYPE(right_basic) == TYPE_GENERIC ||
        TYPE_BASIC_TYPE(left_basic) == TYPE_GNUMERIC_ALL ||
        TYPE_BASIC_TYPE(right_basic) == TYPE_GNUMERIC_ALL){
        goto kind_compatibility;
    }

    if ((TYPE_BASIC_TYPE(left_basic) == TYPE_DREAL &&
         TYPE_BASIC_TYPE(right_basic) == TYPE_REAL) ||
        (TYPE_BASIC_TYPE(left_basic) == TYPE_REAL &&
         TYPE_BASIC_TYPE(right_basic) == TYPE_DREAL)) {
        goto kind_compatibility;
    }

    goto incompatible;

kind_compatibility:

    if (debug_flag) {
        fprintf(debug_fp, "# comparing kind of types\n");
    }

    if (TYPE_BASIC_TYPE(left_basic) == TYPE_DREAL) {
        if (type_is_double(right_basic)) {
            goto length_compatiblity;
        }
    } else if (TYPE_BASIC_TYPE(right_basic) == TYPE_DREAL) {
        if (type_is_double(left_basic)) {
            goto length_compatiblity;
        }
    } else {
        if (TYPE_KIND(left_basic) || TYPE_KIND(right_basic)) {
            if (!type_parameter_expv_equals(
                    TYPE_KIND(left_basic), TYPE_KIND(right_basic), is_strict,
                    for_argunemt, for_assignment, is_pointer_assignment))
                goto incompatible;
        }
    }

length_compatiblity:

    if (debug_flag) {
        fprintf(debug_fp, "# comparing length of types\n");
    }

    if (TYPE_CHAR_LEN(left_basic) > 0 && TYPE_CHAR_LEN(right_basic) > 0) {
        int l1 = TYPE_CHAR_LEN(left_basic);
        int l2 = TYPE_CHAR_LEN(right_basic);
        if (l1 != l2) goto incompatible;
    } else if (TYPE_LENG(left_basic) && TYPE_LENG(right_basic)) {
        if (TYPE_KIND(left_basic) && TYPE_KIND(right_basic)) {
            if (!type_parameter_expv_equals(
                    TYPE_LENG(left_basic), TYPE_LENG(right_basic), is_strict,
                    for_argunemt, for_assignment, is_pointer_assignment))
                goto incompatible;
        }
    }

rank_compatibility:

    if (debug_flag) {
        fprintf(debug_fp, "# comparing rank of types\n");
    }

    if (TYPE_N_DIM(left) == 0 && TYPE_N_DIM(right) == 0) {
        goto attribute_compatibility;

    } else if (TYPE_N_DIM(left) > 0 && TYPE_N_DIM(right) > 0 &&
         are_dimension_and_shape_conformant_by_type(NULL, left, right, NULL, issue_error)) 
    {
        goto attribute_compatibility;
    } else {
        goto incompatible;
    }

attribute_compatibility:

    if (for_assignment || for_argunemt) {
        goto compatible;
    }

    if (debug_flag) {
        fprintf(debug_fp, "# comparing attribute of types\n");
        fprintf(debug_fp, "#  left is '%x', right is '%x'\n",
                TYPE_ATTR_FOR_COMPARE & TYPE_ATTR_FLAGS(left),
                TYPE_ATTR_FOR_COMPARE & TYPE_ATTR_FLAGS(right));
    }

    if (IS_ANY_CLASS(left) || IS_ANY_CLASS(right)) {
        fprintf(debug_fp, "# CLASS(*) \n");
        goto compatible;
    }

    if ((TYPE_ATTR_FOR_COMPARE & TYPE_ATTR_FLAGS(left)) !=
        (TYPE_ATTR_FOR_COMPARE & TYPE_ATTR_FLAGS(right))) {
        goto incompatible;
    }

compatible:
    if (debug_flag) {
        fprintf(debug_fp, "# compatible!\n");
    }

    return TRUE;

incompatible:
    return FALSE;

}


int
type_is_strict_compatible(TYPE_DESC left, TYPE_DESC right, int compare_rank, int issue_error)
{
    return type_is_compatible(left, right,
                              /*is_strict=*/TRUE,
                              /*for_argument=*/FALSE,
                              /*for_assignment=*/FALSE,
                              /*is_pointer_assignment=*/FALSE,
                              /*compare_rank=*/compare_rank,
                              issue_error);
}


static int
type_is_match_for_argument(TYPE_DESC left, TYPE_DESC right, int compare_rank, int issue_error)
{
    return type_is_compatible(left, right,
                              /*is_strict=*/TRUE,
                              /*for_argument=*/TRUE,
                              /*for_assignment=*/FALSE,
                              /*is_pointer_assignment=*/FALSE,
                              /*compare_rank=*/compare_rank,
                              issue_error);
}


int
type_is_compatible_for_allocation(TYPE_DESC left, TYPE_DESC right)
{
    return type_is_compatible(left, right,
                              /*is_strict=*/TRUE,
                              /*for_argument=*/FALSE,
                              /*for_assignment=*/TRUE,
                              /*is_pointer_assignment=*/TRUE,
                              /*compare_rank=*/TRUE,
                              TRUE);
}


static int
function_type_is_compatible0(const TYPE_DESC ftp1, const TYPE_DESC ftp2,
                             int override, int assignment)
{
    ID args1;
    ID args2;
    ID arg1;
    ID arg2;

    /* for type-bound procedure */
    TYPE_DESC tbp1 = NULL;
    TYPE_DESC tbp2 = NULL;

    TYPE_DESC bftp1 = ftp1;
    TYPE_DESC bftp2 = ftp2;

    SYMBOL pass_arg = NULL;

    if (ftp1 == NULL || ftp2 == NULL) {
        // may never reach
        return FALSE;
    }

    if (TYPE_REF(ftp1)) {
        tbp1 = ftp1;
        bftp1 = get_bottom_ref_type(ftp1);
    }
    if (TYPE_REF(ftp2)) {
        tbp2 = ftp2;
        bftp2 = get_bottom_ref_type(ftp2);
    }

    args1 = FUNCTION_TYPE_ARGS(bftp1);
    args2 = FUNCTION_TYPE_ARGS(bftp2);

    /*
     * compare return types
     */
    if (!type_is_strict_compatible(FUNCTION_TYPE_RETURN_TYPE(bftp1),
                                   FUNCTION_TYPE_RETURN_TYPE(bftp2), TRUE, TRUE)) 
    {
        if (debug_flag) {
            fprintf(debug_fp, "return types are not match\n");
        }
        return FALSE;
    }

    if (override || assignment) {
        SYMBOL pass_arg1 = NULL;
        SYMBOL pass_arg2 = NULL;

        if (override) {
            if (tbp1 == NULL || tbp2 == NULL) {
                return FALSE;
            }

            if ((TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tbp1) &&
                 !TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tbp2)) ||
                (!TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tbp1) &&
                 TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tbp2))) {
                return FALSE;
            }
            if (tbp1 != NULL && TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tbp1)) {
                if (TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(tbp1)) {
                    pass_arg1 = ID_SYM(TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(tbp1));
                } else {
                    pass_arg1 = ID_SYM(FUNCTION_TYPE_ARGS(bftp1));
                }
            }
            if (tbp2 != NULL && TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tbp2)) {
                if (TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(tbp2)) {
                    pass_arg2 = ID_SYM(TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(tbp2));
                } else {
                    pass_arg2 = ID_SYM(FUNCTION_TYPE_ARGS(bftp2));
                }
            }
            if (pass_arg1 != pass_arg2) {
                return FALSE;
            }
        } else { /* assignment */
            if (tbp1 == NULL) {
                return FALSE;
            }

            if (FUNCTION_TYPE_HAS_PASS_ARG(tbp1)) {
                if (FUNCTION_TYPE_PASS_ARG(tbp1)) {
                    pass_arg1 = ID_SYM(FUNCTION_TYPE_PASS_ARG(tbp1));
                } else {
                    pass_arg1 = ID_SYM(FUNCTION_TYPE_ARGS(bftp1));
                }
            }

        }
        pass_arg = pass_arg1;
    }

    for (arg1 = args1, arg2 = args2;
         arg1 != NULL && arg2 != NULL;
         arg1 = ID_NEXT(arg1), arg2 = ID_NEXT(arg2)) {

        if (arg1 == NULL || arg2 == NULL) {
            break;
        }

        if (debug_flag) {
            fprintf(debug_fp, "comparing argument '%s' and '%s'\n",
                    SYM_NAME(ID_SYM(arg1)), SYM_NAME(ID_SYM(arg2)));
        }

        if (override) {
            if (ID_SYM(arg1) != ID_SYM(arg2)) {
                return FALSE;
            }
        }

        if (override || assignment) {
            if (pass_arg != NULL && pass_arg == ID_SYM(arg1)) {
                /* when checking override, skip PASS arugment check */
                continue;
            }
        }

        if (!type_is_strict_compatible(ID_TYPE(arg1), ID_TYPE(arg2), TRUE, TRUE)) {
            if (debug_flag) {
                fprintf(debug_fp, "argument types are not match ('%s' and '%s')\n",
                        SYM_NAME(ID_SYM(arg1)), SYM_NAME(ID_SYM(arg2)));
            }
            return FALSE;
        }
    }

    if (arg1 != NULL || arg2 != NULL) {
        /* arugment length are not same */
        if (debug_flag) {
            fprintf(debug_fp, "argument length not match\n");
        }
        return FALSE;
    }

    return TRUE;
}

int
function_type_is_compatible(const TYPE_DESC ftp1, const TYPE_DESC ftp2)
{
    return function_type_is_compatible0(ftp1, ftp2, FALSE, FALSE);
}


/*
 * Check type-bound procedures have the same type except pass arguments.
 */
int
type_bound_procedure_types_are_compatible(const TYPE_DESC tbp1, const TYPE_DESC tbp2)
{
    return function_type_is_compatible0(tbp1, tbp2, TRUE, FALSE);
}


/*
 * Check type-bound procedures have the same type except pass arguments.
 */
int
procedure_pointers_are_compatible(TYPE_DESC p1, TYPE_DESC p2)
{
    return function_type_is_compatible0(p1, p2, FALSE, TRUE);
}


int
procedure_has_pass_arg(const TYPE_DESC ftp, const SYMBOL pass_arg, const TYPE_DESC stp)
{
    ID target;
    TYPE_DESC tp;

    if (ftp == NULL) {
        return FALSE;
    }

    if (!FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(ftp)) {
        return FALSE;
    }

    if (pass_arg == NULL) {
        /* PASS arugment name is not sepcified,
           so first argument become PASS argument */
        target = FUNCTION_TYPE_ARGS(ftp);
    } else {
        target = find_ident_head(pass_arg,
                                 FUNCTION_TYPE_ARGS(ftp));
    }

    if (target == NULL) {
        return FALSE;
    }


    /* check type */
    tp = ID_TYPE(target);
    if (tp == NULL) {
        return FALSE;
    }

    if (type_is_unlimited_class(tp)) {
        return TRUE;

    } else if (type_is_class_of(stp, tp)) {
        return TRUE;

    } else {
        debug("PASS object should be CLASS of the derived-type");
        return FALSE;
    }
}



/**
 * Check PASS of type-bound procedure/procedure variable
 *
 * stp -- the derived-type in which a type-bound procedure or procedure variable appears
 * tbp -- the type of a type-bound procedure or procedure variable
 * ftp -- the function type that is refered from a type-bound procedure or procedure variable
 *
 * The function that is refered from a type-bound procedure or a procedure
 * variable should have a argument that have a CLASS type of the derived-type in
 * which a type-bound procedure or a procedure variable appears.
 *
 * ex)
 *
 *   TYPE t
 *     INTEGER :: v
 *     PROCEDURE(f),PASS(a),POINTER :: p 
 *     ! `p` refer the function `f` that have a arugment `a` and
 *     ! `a` should be a subclass of CLASS(t) or CLASS(*)
 *   END TYPE t
 *
 */
int
check_tbp_pass_arg(TYPE_DESC stp, TYPE_DESC tbp, TYPE_DESC ftp)
{
    ID pass_arg;

    if (!(FUNCTION_TYPE_HAS_PASS_ARG(tbp))) {
        return TRUE;
    }

    pass_arg = FUNCTION_TYPE_PASS_ARG(tbp);

    return procedure_has_pass_arg(ftp, pass_arg?ID_SYM(pass_arg):NULL, stp);
}


int
procedure_is_assignable(const TYPE_DESC left, const TYPE_DESC right)
{
    TYPE_DESC left_ftp;
    TYPE_DESC right_ftp;
    struct type_descriptor type;

    debug("### BEGIN procedure_is_assignable");

    if (left == NULL || right == NULL) {
        debug("#### NULL check fails");
        return FALSE;
    }

    if (!IS_PROCEDURE_TYPE(left) || !IS_PROCEDURE_TYPE(right)) {
        debug("#### Invalid argument, not procedure type");
        return FALSE;
    }

    left_ftp = get_bottom_ref_type(left);

    /* right may be a procedure pointer */
    right_ftp = get_bottom_ref_type(right);

    if (TYPE_IS_EXTERNAL(right_ftp)) {
        type = *right_ftp;
        right_ftp = &type;
        TYPE_UNSET_EXTERNAL(right_ftp);
    }

    if (!TYPE_IS_POINTER(left)) {
        debug("#### Invalid argument, not POINTER");
        /*
         * Left handside operand is not POINTER,
         * so assignment fails.
         */
        return FALSE;
    }

    if (TYPE_IS_NOT_FIXED(left) && IS_SUBR(right_ftp)) {
        /*
         * Left handside operand does not specify any interface,
         * but it expects a function by default.
         * so if right handside operand is subroutine,
         * return TRUE and fix the type in the caller.
         *
         * ex)
         *   PROCEDURE(), POINTER :: left
         *   left => sub() ! success, and left is fixed to a subroutine
         */
        return TRUE;
    }

    if (left == right || left == right_ftp ||
        left_ftp == right || left_ftp == right_ftp) {
        debug("#### The same function");
        /* refers the same function */
        return TRUE;
    }

    if (FUNCTION_TYPE_HAS_PASS_ARG(left)) {
        /*
         * Check right have a PASS argument.
         */
        if (!check_tbp_pass_arg(FUNCTION_TYPE_PASS_ARG_TYPE(left),
                                left, right_ftp)) {
            error("Pointee should have a PASS argument");
            return FALSE;
        }
    }

    if (FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(left_ftp) &&
        FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(right_ftp)) {

        return procedure_pointers_are_compatible(left, right_ftp);

    } else if (!FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(left_ftp)) {

        return type_is_strict_compatible(
            FUNCTION_TYPE_RETURN_TYPE(left_ftp),
            FUNCTION_TYPE_RETURN_TYPE(right_ftp), TRUE, TRUE);

    } else {

        return FALSE;
    }
}


/*
 * Update a function type, decide the type of its arugments.
 */
void
function_type_udpate(TYPE_DESC ftp, ID idList)
{
    ID id;
    ID arg;

    fix_array_dimensions(FUNCTION_TYPE_RETURN_TYPE(ftp));

    FOREACH_ID(arg, FUNCTION_TYPE_ARGS(ftp)) {
        if (ID_TYPE(arg)) {
            continue;
        }
        id = find_ident_head(ID_SYM(arg), idList);
        implicit_declaration(id);
        ID_TYPE(arg) = ID_TYPE(id);
        fix_array_dimensions(ID_TYPE(arg));
    }
}


/*
 * Check the function type is applicable to actual arugments.
 */
/*
 * This function does not consider named argument,
 * so this function is insufficient normal function/subroutine call.
 */
int
function_type_is_appliable(TYPE_DESC ftp, expv actual_args, int issue_error)
{
    expv actual_arg = NULL;
    ID dummy_arg;
    ID dummy_args;
    list actual_lp;
    TYPE_DESC tbp_tp = NULL;
    int compare_rank = TRUE;

    if (ftp == NULL) {
        return FALSE;
    }

    if (TYPE_IS_ELEMENTAL(ftp)) {
        compare_rank = FALSE;
    }

    if (TYPE_REF(ftp) != NULL && FUNCTION_TYPE_IS_TYPE_BOUND(ftp)) {
        /* ftp is the type of type-bound procedure */
        tbp_tp = ftp;
        ftp = TYPE_REF(tbp_tp);
        if (ftp == NULL) {
            /* error("type-bound procedure "); */
            return FALSE;
        }
    }

    dummy_args = FUNCTION_TYPE_ARGS(ftp);

    actual_lp = EXPR_LIST(actual_args);
    actual_arg = actual_lp ? LIST_ITEM(actual_lp) : NULL;

    FOREACH_ID(dummy_arg, dummy_args) {

        if (debug_flag)
            fprintf(debug_fp, "dummy args is '%s'\n", SYM_NAME(ID_SYM(dummy_arg)));

        /* skip type check for PASS argument */
        if (tbp_tp != NULL && TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tbp_tp)) {
            if (TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(tbp_tp)) {
                if (ID_SYM(dummy_arg) == ID_SYM(TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(tbp_tp))) {
                    continue;
                }
            } else {
                /* If the PASS argument is not specified, the first arugment is the PASS argument*/
                if (dummy_arg == dummy_args) {
                    continue;
                }
            }
        }

        if (actual_arg == NULL) {
            /* argument number mismatch */
            return FALSE;
        }

        if (!isValidType(EXPV_TYPE(actual_arg))) {
            return FALSE;
        }

        if (!type_is_match_for_argument(EXPV_TYPE(actual_arg),
                                       ID_TYPE(dummy_arg), compare_rank, issue_error))
        {
            return FALSE;
        }

        if (LIST_NEXT(actual_lp) != NULL) {
            actual_lp = LIST_NEXT(actual_lp);
            actual_arg = LIST_ITEM(actual_lp);
        } else {
            actual_lp = NULL;
            actual_arg = NULL;
        }
    }

    if (actual_lp != NULL) {
        return FALSE;
    }

    return TRUE;
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

    btp2 = getBaseParameterizedType(tp2);

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
        if (derived_type_parameter_values_is_compatible_for_assignment(tp1, tp2, is_pointer_set)) {
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

    if (!isValidType(tp1) || !isValidType(tp2)) {
        return FALSE;
    }

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
