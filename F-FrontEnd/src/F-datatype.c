/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
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
    while (TYPE_REF(td) && IS_STRUCT_TYPE(TYPE_REF(td))) {
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
    ID member;

    if (!IS_STRUCT_TYPE(struct_td)) {
        return NULL;
    }
    struct_td = getBaseStructType(struct_td);
    FOREACH_MEMBER(member, struct_td) {
        if (strcmp(ID_NAME(member), SYM_NAME(sym)) == 0) {
            return member;
        }
    }
    return NULL;
}

int
is_descendant_coindexed(TYPE_DESC tp){

  ID id;

  if (!tp) return FALSE;

  if (TYPE_IS_COINDEXED(tp)) return TRUE;

  if (IS_STRUCT_TYPE(tp)){

    FOREACH_MEMBER(id, tp){
      if (is_descendant_coindexed(ID_TYPE(id))) return TRUE;
    }

    if (TYPE_REF(tp)) return is_descendant_coindexed(TYPE_REF(tp));

  }
  else if (IS_ARRAY_TYPE(tp)){
    return is_descendant_coindexed(bottom_type(tp));
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
        if (b2 == TYPE_STRUCT || b2 == TYPE_GENERIC)
            return TRUE;
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
