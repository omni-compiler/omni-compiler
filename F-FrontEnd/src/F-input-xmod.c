/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-input-xmod.c
 */

#include <libxml/xmlreader.h>
#include "F-front.h"
#include "F-input-xmod.h"
#include "hash.h"

typedef struct type_entry {
    int hasExtID;
    EXT_ID ep;
    TYPE_DESC tp;
} * TYPE_ENTRY;

static int input_FfunctionDecl(xmlTextReaderPtr, HashTable *, EXT_ID, ID);

static int
xmlSkipWhiteSpace(xmlTextReaderPtr reader) 
{
    int type;

    do {
        if (xmlTextReaderRead(reader) != 1) {
            fprintf(stderr, "could not read xml file further.\n");
            fflush(stderr);
            return FALSE;
        }

        type = xmlTextReaderNodeType(reader);
    } while (type == XML_READER_TYPE_COMMENT ||
             type == XML_READER_TYPE_WHITESPACE ||
             type == XML_READER_TYPE_SIGNIFICANT_WHITESPACE);

    return TRUE;
}

static int
xmlSkipUntil(xmlTextReaderPtr reader, int type, const char * name, int depth)
{
    int current_type;
    int current_depth;
    const char * current_name;

    do {
        if (!xmlTextReaderRead(reader))
            return FALSE;

            current_type = xmlTextReaderNodeType(reader);
            current_name = (const char *) xmlTextReaderConstName(reader);
            current_depth = xmlTextReaderDepth(reader);
    } while (current_type != type ||
             strcmp(current_name, name) != 0 ||
             current_depth != depth);

    return TRUE;
}

static int
xmlMatchNodeType(xmlTextReaderPtr reader, int type)
{
    return (xmlTextReaderNodeType(reader) == type);
}

#if 0
static int
xmlExpectNodeType(xmlTextReaderPtr reader, int type)
{
    if (!xmlExpectNodeType(reader, type)) {
        fprintf(stderr, "expected node type %d, but was %d.\n",
            type, xmlTextReaderNodeType(reader));
        fflush(stderr);
        return FALSE;
    }

    xmlSkipWhiteSpace(reader);

    return TRUE;
}
#endif

static int
xmlMatchNode(xmlTextReaderPtr reader, int type, const char* name)
{
    if (!xmlMatchNodeType(reader, type))
        return FALSE;

    if (strcmp((const char *) xmlTextReaderConstName(reader) , name) != 0)
        return FALSE;

    return TRUE;
}

static int
xmlExpectNode(xmlTextReaderPtr reader, int type, const char* name)
{
    if (!xmlMatchNode(reader, type, name)) {
        fprintf(stderr, "expected node type: %d, name: %s, "
                        "but wad type: %d, name: %s\n",
                        type, name,
                        xmlTextReaderNodeType(reader),
                        xmlTextReaderConstName(reader));
        fflush(stderr);
        return FALSE;
    }

    /* move to next node */
    xmlSkipWhiteSpace(reader);

    return TRUE;
}

static TYPE_ENTRY
getTypeEntry(HashTable * ht, const char * typeId) {
    TYPE_ENTRY tep = NULL;
    TYPE_DESC tp = NULL;
    HashEntry * e;
    int isNew = 0;

    e = FindHashEntry(ht, typeId);
    if (e != NULL) {
        tep = GetHashValue(e);
    } else {
        tep = XMALLOC(TYPE_ENTRY, sizeof(*tep));
        tp = new_type_desc();
        TYPE_IS_DECLARED(tp) = TRUE;
        tep->tp = tp;
        if (strcmp(typeId, "Fint") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_INT;
        } else if (strcmp(typeId, "Fcharacter") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_CHAR;
            TYPE_CHAR_LEN(tp) = 1;
        } else if (strcmp(typeId, "Freal") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_REAL;
        } else if (strcmp(typeId, "Fnumeric") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_GNUMERIC;
        } else if (strcmp(typeId, "FnumericAll") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_GNUMERIC_ALL;
        } else if (strcmp(typeId, "Fcomplex") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_COMPLEX;
        } else if (strcmp(typeId, "Flogical") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_LOGICAL;
        } else if (strcmp(typeId, "Fnamelist") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_NAMELIST;
        } else {
            e = CreateHashEntry(ht, typeId, &isNew);
            SetHashValue(e, tep);
        }
    }
    return tep;
}

static TYPE_DESC
getTypeDesc(HashTable * ht, const char * typeId) {
    TYPE_ENTRY tep = NULL;

    tep = getTypeEntry(ht, typeId);
    
    return tep->tp;
}

#if 0
static void
updateTypeDesc(HashTable * ht, const char * typeId, TYPE_DESC tp) {
    HashEntry * e;
    TYPE_ENTRY tep = NULL;

    e = FindHashEntry(ht, typeId);
    if (e != NULL) {
        tep = GetHashValue(e);
        tep->tp = tp;
    }
}
#endif

static void
setReturnType(HashTable * ht, TYPE_DESC tp, const char * rtid)
{
    if (strcmp(rtid, "Fvoid") == 0) {
        TYPE_BASIC_TYPE(tp) = TYPE_SUBR;
    } else if (strncmp(rtid, "V", 1) == 0) {
        TYPE_BASIC_TYPE(tp) = TYPE_GENERIC;
        TYPE_REF(tp) = getTypeDesc(ht, rtid);
    } else {
        TYPE_BASIC_TYPE(tp) = TYPE_FUNCTION;
        TYPE_REF(tp) = getTypeDesc(ht, rtid);
    }
}

/**
 * input <name> node
 */
static int
input_name(xmlTextReaderPtr reader, SYMBOL * s)
{
    const char * name;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name"))
       return FALSE;

    if (!xmlMatchNodeType(reader, XML_READER_TYPE_TEXT))
       return FALSE;

    name = (const char*) xmlTextReaderConstValue(reader);
    *s = find_symbol(name);

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name"))
        return FALSE;

    return TRUE;
}

/**
 * input type and attribute at node
 */
static int
input_type_and_attr(xmlTextReaderPtr reader, HashTable * ht, char ** typeId,
                    TYPE_DESC * tp)
{
    char * str;

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "type");
    if (str == NULL)
        return FALSE;

    *tp = getTypeDesc(ht, str);

    if (typeId != NULL)
        *typeId = str;    /* return typeId */
    else
        free(str);

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "intent");
    if (str != NULL) {
        if (strcmp(str, "in") == 0)
            TYPE_SET_INTENT_IN(*tp);
        else if (strcmp(str, "out") == 0)
            TYPE_SET_INTENT_OUT(*tp);
        else if (strcmp(str, "inout") == 0)
            TYPE_SET_INTENT_INOUT(*tp);

        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_pointer");
    if (str != NULL) {
        TYPE_SET_POINTER(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_target");
    if (str != NULL) {
        TYPE_SET_TARGET(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_optional");
    if (str != NULL) {
        TYPE_SET_OPTIONAL(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_save");
    if (str != NULL) {
        TYPE_SET_SAVE(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_parameter");
    if (str != NULL) {
        TYPE_SET_PARAMETER(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader,
                        BAD_CAST "is_allocatable");
    if (str != NULL) {
        TYPE_SET_ALLOCATABLE(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_sequence");
    if (str != NULL) {
        TYPE_SET_SEQUENCE(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader,
                        BAD_CAST "is_internal_private");
    if (str != NULL) {
        TYPE_SET_INTERNAL_PRIVATE(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_intrinsic");
    if (str != NULL) {
        TYPE_SET_INTRINSIC(*tp);
        free(str);
    }

    return TRUE;
}

/**
 * input <name> node with type attribute
 */
static int
input_name_with_type(xmlTextReaderPtr reader, HashTable * ht,
                     int type_is_optional, SYMBOL * s, TYPE_DESC * tp)
{
    const char * name;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "name"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, NULL, tp)) {
        if (!type_is_optional)
            return FALSE;
    }

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!xmlMatchNodeType(reader, XML_READER_TYPE_TEXT))
       return FALSE;

    name = (const char*) xmlTextReaderConstValue(reader);
    *s = find_symbol(name);

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name"))
        return FALSE;

    return TRUE;
}

/**
 * input <depends> node
 */
static int
input_depends(xmlTextReaderPtr reader, struct module * mod) {
    SYMBOL name;
    struct depend_module * depmod;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "depends"))
        return FALSE;

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </depends> */
            break;

        if (!input_name(reader, &name))
            return FALSE;

        depmod = XMALLOC(struct depend_module *, sizeof(struct depend_module));
        depmod->module_name = name;

        if (mod->depend.head == NULL) {
             mod->depend.head = depmod;
             mod->depend.last = depmod;
        } else {
             mod->depend.last->next = depmod;
             mod->depend.last = depmod;
        }
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "depends"))
        return FALSE;

    return TRUE;
}

/**
 * input expv node
 */
static int
input_expv(xmlTextReaderPtr reader, expv * v)
{
    const char * name;
    int depth;

    if (!xmlMatchNodeType(reader, XML_READER_TYPE_ELEMENT))
        return FALSE;

    if (!xmlTextReaderIsEmptyElement(reader)) {
        name = (const char *) xmlTextReaderConstName(reader);
        depth = xmlTextReaderDepth(reader);

        /* skip until corresponding close tag */
        if (!xmlSkipUntil(reader, XML_READER_TYPE_END_ELEMENT, name, depth))
            return FALSE;
    }

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    /* create dummy expv */
    *v = list0(F_MODULE_INTERNAL);
    EXPV_TYPE(*v) = type_basic(TYPE_INT);

    return TRUE;
}

/**
 * input <kind> node
 */
static int
input_kind(xmlTextReaderPtr reader, TYPE_DESC tp)
{
    expv v;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "kind"))
        return TRUE;

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!input_expv(reader, &v))
        return FALSE;

    TYPE_KIND(tp) = v;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "kind"))
        return FALSE;

    return TRUE;
}

/**
 * input <len> node
 */
static int
input_len(xmlTextReaderPtr reader, TYPE_DESC tp)
{
    expv v = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "len"))
        return TRUE;

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "len")) {
        /* if <len> tag is empty, size is unfixed */
        TYPE_CHAR_LEN(tp) = CHAR_LEN_UNFIXED;
    } else {
        if (!input_expv(reader, &v))
            return FALSE;

        if (v != NULL)
            TYPE_LENG(tp) = v;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "len"))
        return FALSE;

    return TRUE;
}

/**
 * input <lowerBound> node
 */
static int
input_lowerBound(xmlTextReaderPtr reader, expv * v)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "lowerBound"))
        return FALSE;

    if (!input_expv(reader, v))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "lowerBound"))
        return FALSE;

    return TRUE;
}

/**
 * input <upperBound> node
 */
static int
input_upperBound(xmlTextReaderPtr reader, expv * v)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "upperBound"))
        return FALSE;

    if (!input_expv(reader, v))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "upperBound"))
        return FALSE;

    return TRUE;
}

/**
 * input <step> node
 */
static int
input_step(xmlTextReaderPtr reader, expv * v)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "step"))
        return FALSE;

    if (!input_expv(reader, v))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "step"))
        return FALSE;

    return TRUE;
}

/**
 * input <indexRange> node
 */
static int
input_indexRange(xmlTextReaderPtr reader, TYPE_DESC tp)
{
    TYPE_DESC bottom, base;
    expv v;
    char * is_assumed_size;
    char * is_assumed_shape;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "indexRange"))
        return FALSE;

    bottom = tp;
    while(TYPE_BASIC_TYPE(bottom) == TYPE_ARRAY) {
        TYPE_N_DIM(bottom)++;
        bottom = TYPE_REF(bottom);
    }

    base = new_type_desc();
    *base = *bottom;
    TYPE_BASIC_TYPE(bottom) = TYPE_ARRAY;
    TYPE_REF(bottom) = base;
    TYPE_N_DIM(bottom)++;

    /* fix allocatable attribute set in input_FbasicType() */
    if (TYPE_IS_ALLOCATABLE(bottom)) {
        TYPE_SET_ALLOCATABLE(base);
    }

    is_assumed_size = (char *) xmlTextReaderGetAttribute(reader,
                                   BAD_CAST "is_assumed_size");

    is_assumed_shape = (char *) xmlTextReaderGetAttribute(reader,
                                    BAD_CAST "is_assumed_shape");

    if (is_assumed_size != NULL) {
        TYPE_ARRAY_ASSUME_KIND(bottom) = ASSUMED_SIZE;
        free(is_assumed_size);
    } else if (is_assumed_shape != NULL) {
        TYPE_ARRAY_ASSUME_KIND(bottom) = ASSUMED_SHAPE;
        free(is_assumed_shape);
    } else {
        TYPE_ARRAY_ASSUME_KIND(bottom) = ASSUMED_NONE;
    }

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "lowerBound")) {
        if (input_lowerBound(reader, &v))
            TYPE_DIM_LOWER(bottom) = v;
        else
            return FALSE;
    }

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "upperBound")) {
        if (input_upperBound(reader, &v))
            TYPE_DIM_UPPER(bottom) = v;
        else
            return FALSE;
    }

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "step")) {
        if (input_step(reader, &v))
            TYPE_DIM_STEP(bottom) = v;
        else
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "indexRange"))
        return FALSE;

    return TRUE;
}

/**
 * input <indexRange> node in <coShape> node
 */
static int
input_indexRange_coShape(xmlTextReaderPtr reader, expv lp)
{
    expv cobound = NULL;
    expv lower = NULL;
    expv upper = NULL;
    expv step = NULL;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "indexRange"))
        return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "lowerBound"))
        if (!input_lowerBound(reader, &lower))
            return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "upperBound"))
        if (!input_upperBound(reader, &upper))
            return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "step"))
        if (!input_step(reader, &step))
            return FALSE;

    cobound = list3(LIST, lower, upper, step);
    list_put_last(lp, cobound);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "indexRange"))
        return FALSE;

    return TRUE;
}

/**
 * input <coShape> node
 */
static int
input_coShape(xmlTextReaderPtr reader, TYPE_DESC tp)
{
    codims_desc * codims;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "coShape"))
        return FALSE;

    codims = XMALLOC(codims_desc *, sizeof(*codims));
    tp->codims = codims;
    codims->cobound_list = EMPTY_LIST;

    while (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "indexRange")) {
        if (!input_indexRange_coShape(reader, codims->cobound_list))
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "coShape"))
        return FALSE;

    return TRUE;
}

/**
 * input <FbasicType> node
 */
static int
input_FbasicType(xmlTextReaderPtr reader, HashTable * ht)
{
    TYPE_DESC tp = NULL;
    char * typeId = NULL;
    char * ref;
    int isEmpty;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FbasicType"))
        return FALSE;

    isEmpty = xmlTextReaderIsEmptyElement(reader);

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    ref = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "ref");
    TYPE_REF(tp) = getTypeDesc(ht, ref);
    TYPE_BASIC_TYPE(tp) = TYPE_BASIC_TYPE(TYPE_REF(tp));
    shrink_type(tp);

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (isEmpty)
        return TRUE;	/* case FbasicType node has no child */

    /* <kind> */
    if (!input_kind(reader, TYPE_REF(tp)))    /* kind should be set to ref */
        return FALSE;

    /* <len> */
    if (!input_len(reader, tp))
        return FALSE;

    /* <indexRange> */
    while (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "indexRange")) {
        if (!input_indexRange(reader, tp))
            return FALSE;
    }

    /* <coShape> */
    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "coShape"))
        if (!input_coShape(reader, tp))
            return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FbasicType"))
        return FALSE;

    if (typeId != NULL)
        free(typeId);

    return TRUE;
}

/**
 * input <name> node with type attribute appears in <params>
 */
static int
input_param(xmlTextReaderPtr reader, HashTable * ht, EXT_ID ep)
{
    TYPE_DESC tp = NULL;
    SYMBOL s;
    expv v;

    if (!input_name_with_type(reader, ht, TRUE, &s, &tp))
        return FALSE;

    /* note: when type is ENTRY, type is not specified in xmod */
    if (tp == NULL)
        EXT_PROC_CLASS(ep) = EP_ENTRY;

    v = XMALLOC(expv, sizeof(*v));
    EXPV_CODE(v) = IDENT;
    EXPV_TYPE(v) = tp;
    EXPV_NAME(v) = s;
    v = list1(LIST, v);

    list_put_last(EXT_PROC_ARGS(ep), v);

    return TRUE;
}

/**
 * input <FfunctionType> node
 */
static int
input_FfunctionType(xmlTextReaderPtr reader, HashTable * ht)
{
    TYPE_ENTRY tep;
    char * typeId;
    char * attr;
    int isEmpty;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FfunctionType"))
        return FALSE;

    isEmpty = xmlTextReaderIsEmptyElement(reader);

    typeId = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "type");
    tep = getTypeEntry(ht, typeId);
    tep->hasExtID = TRUE;
    /* SYMBOL is set later in <id> */
    tep->ep = new_external_id(NULL);
    EXT_PROC_ARGS(tep->ep) = EMPTY_LIST;
    EXT_IS_DEFINED(tep->ep) = TRUE;
    EXT_TAG(tep->ep) = STG_EXT;

    setReturnType(ht, tep->tp, (char *) xmlTextReaderGetAttribute(reader,
                               BAD_CAST "return_type"));

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_program");
    if (attr != NULL) {
        EXT_PROC_CLASS(tep->ep) = EP_PROGRAM;
        free(attr);
    }

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_intrinsic");
    if (attr != NULL) {
        EXT_PROC_CLASS(tep->ep) = EP_INTRINSIC;
        free(attr);
    }
    
    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_recursive");
    if (attr != NULL) {
        TYPE_SET_RECURSIVE(tep->tp);
        free(attr);
    }

#ifdef SUPPORT_PURE
    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_pure");
    if (attr != NULL) {
        TYPE_SET_PURE(tep->tp);
        free(attr);
    }
#endif

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_external");
    if (attr != NULL) {
        TYPE_SET_EXTERNAL(tep->tp);
        free(attr);
    }

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (isEmpty)
        return TRUE;	/* case FfunctionType node has no child */

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "params"))
        return FALSE;

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </params> */
            break;

        if (!input_param(reader, ht, tep->ep))
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "params")) 
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FfunctionType")) 
        return FALSE;

    return TRUE;
}

/**
 * input <id> node which appears in <symbols>
 */
static int
input_symbol(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC parent, ID * tail)
{
    ID id;
    SYMBOL s;
    TYPE_DESC tp;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "id"))
        return FALSE;
    
    if (!input_type_and_attr(reader, ht, NULL, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_name(reader, &s))
        return FALSE;

    id = new_ident_desc(s);
    ID_TYPE(id) = tp;
    if(type_is_omissible(tp, 0, 0))
        ID_TYPE(id) = TYPE_REF(tp);

    ID_LINK_ADD(id, TYPE_MEMBER_LIST(parent), *tail);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "id"))
        return FALSE;

    return TRUE;
}

/**
 * input <FstructType> node
 */
static int
input_FstructType(xmlTextReaderPtr reader, HashTable * ht)
{
    TYPE_DESC tp;
    ID tail = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FstructType"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, NULL, &tp))
        return FALSE;

    TYPE_BASIC_TYPE(tp) = TYPE_STRUCT;

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "symbols"))
        return FALSE;

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </symbols> */
            break;

        if (!input_symbol(reader, ht, tp, &tail))
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "symbols"))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FstructType"))
        return FALSE;

    return TRUE;
}

/**
 * input <typeTable> node
 */
static int
input_typeTable(xmlTextReaderPtr reader, HashTable * ht)
{
    const char * name;
    int succeeded;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "typeTable"))
        return FALSE;

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </typeTable> */
            break;

        name = (const char *)xmlTextReaderConstName(reader);
        if (strcmp(name, "FbasicType") == 0) {
            succeeded = input_FbasicType(reader, ht);
        } else if (strcmp(name, "FfunctionType") == 0) {
            succeeded = input_FfunctionType(reader, ht);
        } else if (strcmp(name, "FstructType") == 0) {
            succeeded = input_FstructType(reader, ht);
        } else {
            fprintf(stderr, "Unknown type entry in xmod file: %s.\n", name);
            fflush(stderr);
            return FALSE;
        }

        if (!succeeded)
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "typeTable"))
        return FALSE;

    return TRUE;
}

static void
set_sclass(ID id, const char* sclass)
{
    TYPE_DESC tp = ID_TYPE(id);
    if (tp != NULL) {
        if (TYPE_IS_PARAMETER(tp))
            ID_CLASS(id) = CL_PARAM;
        else
            ID_CLASS(id) = CL_VAR;

        if (IS_FUNCTION_TYPE(tp) || IS_SUBR(tp) || IS_GENERIC_TYPE(tp)) {
            ID_CLASS(id) = CL_PROC;
            /* ID_STORAGE(id) = STG_SAVE; */
        }
    }

    if (sclass == NULL)
        return;

    if (strcmp(sclass, "flocal") == 0)
        ID_STORAGE(id) = STG_AUTO;
    else if (strcmp(sclass, "fcommon_name") == 0)
        ID_CLASS(id) = CL_COMMON;
    else if (strcmp(sclass, "fnamelist_name") == 0)
        ID_CLASS(id) = CL_NAMELIST;
    else if (strcmp(sclass, "fparam") == 0)
        ID_STORAGE(id) = STG_ARG;
    else if (strcmp(sclass, "ffunc") == 0)
        ID_STORAGE(id) = STG_EXT;
    else if (strcmp(sclass, "fcommon") == 0)
        ID_STORAGE(id) = STG_COMMON;
    else if (strcmp(sclass, "fsave") == 0)
        ID_STORAGE(id) = STG_SAVE;
    else if (strcmp(sclass, "ftype_name") == 0) {
        ID_STORAGE(id) = STG_TAGNAME;
        ID_CLASS(id) = CL_TAGNAME;
    }
}

/**
 * input <id> node
 */
static int
input_id(xmlTextReaderPtr reader, HashTable * ht, struct module * mod)
{
    char * type;
    char * original_name;
    char * declared_in;
    char * is_ambiguous;
    char * sclass;
    TYPE_ENTRY tep;
    SYMBOL name;
    ID id;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "id"))
       return FALSE;

    type = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "type");
    original_name = (char *) xmlTextReaderGetAttribute(reader,
                                 BAD_CAST "original_name");
    declared_in = (char *) xmlTextReaderGetAttribute(reader,
                               BAD_CAST "declared_in");
    is_ambiguous = (char *) xmlTextReaderGetAttribute(reader,
                               BAD_CAST "is_ambiguous");
    sclass = (char *) xmlTextReaderGetAttribute(reader,
                          BAD_CAST "sclass");

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!input_name(reader, &name))
        return FALSE;

    id = XMALLOC(ID, sizeof(*id));
    ID_SYM(id) = name;
    id->use_assoc = XMALLOC(struct use_assoc_info *, sizeof(*id->use_assoc));
    id->use_assoc->original_name = find_symbol(original_name);
    id->use_assoc->module_name = find_symbol(declared_in);
    if (is_ambiguous != NULL && strcmp("true", is_ambiguous) == 0) {
        ID_IS_AMBIGUOUS(id) = TRUE;
    } else {
        ID_IS_AMBIGUOUS(id) = FALSE;
    }

    if (type != NULL) {
        tep = getTypeEntry(ht, type);
        ID_TYPE(id) = tep->tp;
        if (tep->hasExtID) {
            PROC_EXT_ID(id) = tep->ep;
            EXT_SYM(tep->ep) = name;
            EXT_PROC_TYPE(tep->ep) = ID_TYPE(id);
        } else if (type_is_omissible(ID_TYPE(id), 0, 0)) {
            ID_TYPE(id) = TYPE_REF(ID_TYPE(id));
        }

        // if type of id is function/subroutine, then regarded as procedure
        if (TYPE_BASIC_TYPE(ID_TYPE(id)) == TYPE_FUNCTION ||
            TYPE_BASIC_TYPE(ID_TYPE(id)) == TYPE_SUBR ||
            TYPE_BASIC_TYPE(ID_TYPE(id)) == TYPE_GENERIC) {
            ID_IS_DECLARED(id) = TRUE;
            if (TYPE_IS_EXTERNAL(ID_TYPE(id)))
                PROC_CLASS(id) = P_EXTERNAL;
            else
                PROC_CLASS(id) = P_DEFINEDPROC;
            ID_ADDR(id) = expv_sym_term(F_FUNC, ID_TYPE(id), ID_SYM(id));

            // only for type of <functionCall/>
            ID_DEFINED_BY(id) = new_ident_desc(ID_SYM(id));
            *ID_DEFINED_BY(id) = *id;
            ID_TYPE(ID_DEFINED_BY(id)) = TYPE_REF(ID_TYPE(id));

            /* case for ENTRY */
            if (EXT_PROC_IS_ENTRY(tep->ep))
                ID_CLASS(id) = CL_ENTRY;
        }
    }

    set_sclass(id, sclass);

    if (mod->last == NULL) {
        mod->last = id;
        mod->head = id;
    } else {
        ID_NEXT(mod->last) = id;
        mod->last = id;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "id"))
        return FALSE;

    free(type);
    free(original_name);
    free(declared_in);
    if (sclass != NULL)
        free(sclass);

    return TRUE;
}

/**
 * input <identifiers> node
 */
static int
input_identifiers(xmlTextReaderPtr reader, HashTable * ht, struct module * mod)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "identifiers"))
        return FALSE;

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </identifiers> */
            break;

        if (!input_id(reader, ht, mod))
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "identifiers"))
        return FALSE;

    return TRUE;
}

/**
 * input <FmoduleProcedureDecl> node
 */
static int
input_FmoduleProcedureDecl(xmlTextReaderPtr reader, HashTable *ht,
                           EXT_ID parent)
{
    SYMBOL s;
    TYPE_ENTRY tep;
    char *typeId;
    char *name;
    int ret = TRUE;

    if (!xmlExpectNode(reader,
                       XML_READER_TYPE_ELEMENT, "FmoduleProcedureDecl"))
        return FALSE;

    while (!xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT) &&
           ret == TRUE) {
        tep = NULL;
        typeId = NULL;
        name = NULL;
        s = NULL;

        if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "name")) {
            ret = FALSE;
            goto loopEnd;
        }

        typeId = (char *)xmlTextReaderGetAttribute(reader, BAD_CAST "type");
        if (!xmlSkipWhiteSpace(reader)) {
            ret = FALSE;
            goto loopEnd;
        }

        name = (char *)xmlTextReaderConstValue(reader);
        if (name != NULL) {
            name = strdup(name);
        }
        if (!xmlSkipWhiteSpace(reader)) {
            ret = FALSE;
            goto loopEnd;
        }

        if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name")) {
            ret = FALSE;
            goto loopEnd;
        }

        if (typeId != NULL &&
            (tep = getTypeEntry(ht, typeId)) != NULL &&
            tep->hasExtID == TRUE &&
            tep->ep != NULL) {
            
            s = find_symbol(name);
            EXT_SYM(tep->ep) = s;

            if (EXT_PROC_TYPE(tep->ep) == NULL) {
                EXT_PROC_TYPE(tep->ep) = tep->tp;
            }

            if (EXT_PROC_INTR_DEF_EXT_IDS(parent) == NULL)
                EXT_PROC_INTR_DEF_EXT_IDS(parent) = tep->ep;
            else
                extid_put_last(EXT_PROC_INTR_DEF_EXT_IDS(parent), tep->ep);
        } else {
            ret = FALSE;
        }

        loopEnd:
        free(typeId);
        free(name);
    }

    if (!xmlExpectNode(reader,XML_READER_TYPE_END_ELEMENT,
                       "FmoduleProcedureDecl"))
        return FALSE;

    return ret;
}

/**
 * input <value> node
 */
static int
input_value(xmlTextReaderPtr reader, ID id)
{
    expv v = NULL;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "value"))
        return FALSE;

    if (!input_expv(reader, &v))
        return FALSE;

    VAR_INIT_VALUE(id) = v;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "value"))
        return FALSE;

    return TRUE;
}

/**
 * input <varDecl> node
 */
static int
input_varDecl(xmlTextReaderPtr reader, HashTable * ht, EXT_ID parent)
{
    SYMBOL s;
    TYPE_DESC tp;
    ID id;
    ID tail = NULL;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "varDecl"))
        return FALSE;

    if (!input_name_with_type(reader, ht, FALSE, &s, &tp))
        return FALSE;

    id = XMALLOC(ID, sizeof(*id));
    ID_SYM(id) = s;
    ID_TYPE(id) = tp;

    ID_LINK_ADD(id, EXT_PROC_ID_LIST(parent), tail);

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "value"))
        if (!input_value(reader, id))
            return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "varDecl"))
        return FALSE;

    return TRUE;
}

/**
 * input <FuseDecl> node
 */
static int
input_FuseDecl(xmlTextReaderPtr reader)
{
    int depth;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FuseDecl"))
        return FALSE;

    if (!xmlTextReaderIsEmptyElement(reader)) {
        depth = xmlTextReaderDepth(reader);

        /* skip until corresponding close tag */
        if (!xmlSkipUntil(reader, XML_READER_TYPE_END_ELEMENT,"FuseDecl",
                          depth))
            return FALSE;
    }

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    return TRUE;
}

/**
 * input <FuseOnlyDecl> node
 */
static int
input_FuseOnlyDecl(xmlTextReaderPtr reader)
{
    int depth;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FuseOnlyDecl"))
        return FALSE;

    if (!xmlTextReaderIsEmptyElement(reader)) {
        depth = xmlTextReaderDepth(reader);

        /* skip until corresponding close tag */
        if (!xmlSkipUntil(reader, XML_READER_TYPE_END_ELEMENT,"FuseOnlyDecl",
                          depth))
            return FALSE;
    }

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    return TRUE;
}

/**
 * input <FinterfaceDecl> node in Ffunctiondecl/declarations
 */
static int
input_FinterfaceDecl_in_declarations(xmlTextReaderPtr reader, HashTable * ht,
                                     EXT_ID parent, ID id_list)
{
    EXT_ID ep;
    char * name = NULL;
    char * is_operator = NULL;
    char * is_assignment = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FinterfaceDecl"))
        return FALSE;

    name = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "name");
    is_operator = (char *) xmlTextReaderGetAttribute(reader,
                               BAD_CAST "is_operator");
    is_assignment = (char *) xmlTextReaderGetAttribute(reader,
                                 BAD_CAST "is_assignment");

    if (name != NULL) {
        ep = new_external_id(find_symbol(name));
        free(name);
    } else if (is_assignment != NULL) {
        ep = new_external_id(find_symbol("="));
    } else {
        ep = new_external_id(NULL);
        EXT_IS_BLANK_NAME(ep) = TRUE;
    }

    if (is_operator != NULL) {
        EXT_PROC_INTERFACE_CLASS(ep) = INTF_OPERATOR;
        free(is_operator);
    }

    if (is_assignment != NULL) {
        EXT_PROC_INTERFACE_CLASS(ep) = INTF_ASSINGMENT;
        free(is_assignment);
    }

    EXT_PROC_INTERFACE_INFO(ep) =
        XMALLOC(struct interface_info *, sizeof(struct interface_info));
    EXT_IS_BLANK_NAME(ep) = FALSE;
    EXT_PROC_CLASS(ep) = EP_INTERFACE;
    EXT_PROC_INTERFACE_CLASS(ep) = INTF_DECL;

    if (EXT_PROC_INTERFACES(parent) == NULL)
        EXT_PROC_INTERFACES(parent) = ep;
    else
        extid_put_last(EXT_PROC_INTERFACES(parent), ep);

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    while (!xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT)) {
        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                         "FmoduleProcedureDecl")) {
            if (!input_FmoduleProcedureDecl(reader, ht, ep))
                return FALSE;
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                                "FfunctionDecl")) {
            if (!input_FfunctionDecl(reader, ht, ep, id_list))
                return FALSE;
        }
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FinterfaceDecl"))
        return FALSE;

    return TRUE;
}

/**
 * input <declarations> node
 */
static int
input_declarations(xmlTextReaderPtr reader, HashTable * ht, EXT_ID parent,
                   ID id_list)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "declarations"))
        return FALSE;

    while (!xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT)) {
        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "varDecl")) {
            if (!input_varDecl(reader, ht, parent))
                return FALSE;
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FuseDecl")) {
            if (!input_FuseDecl(reader))
                return FALSE;
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                                "FuseOnlyDecl")) {
            if (!input_FuseOnlyDecl(reader))
                return FALSE;
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                                "FinterfaceDecl")) {
            if (!input_FinterfaceDecl_in_declarations(reader, ht, parent,
                                                      id_list))
                return FALSE;
        } else {
            fprintf(stderr, "unexpected node: %s in <declarations> node.\n",
                    (const char *)xmlTextReaderConstName(reader));
            fflush(stderr);
            return FALSE;
        }
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "declarations"))
        return FALSE;

    return TRUE;
}

/**
 * input <FfunctionDecl> node
 */
static int
input_FfunctionDecl(xmlTextReaderPtr reader, HashTable * ht, EXT_ID parent,
                    ID id_list)
{
    ID id;
    EXT_ID ep;
    SYMBOL s;
    TYPE_DESC tp = NULL;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "FfunctionDecl"))
        return FALSE;

    if (!input_name_with_type(reader, ht, TRUE, &s, &tp))
        return FALSE;

    id = find_ident_head(s, id_list); 
    if (id != NULL) {
        ep = PROC_EXT_ID(id);
    } else {
        ep = new_external_id(s);
        EXT_PROC_TYPE(ep) = tp;
    }

    if (EXT_PROC_INTR_DEF_EXT_IDS(parent) == NULL)
        EXT_PROC_INTR_DEF_EXT_IDS(parent) = ep;
    else
        extid_put_last(EXT_PROC_INTR_DEF_EXT_IDS(parent), ep);

    if (!input_declarations(reader, ht, ep, id_list))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FfunctionDecl"))
        return FALSE;

    return TRUE;
}

/**
 * input <FinterfaceDecl> node
 */
static int
input_FinterfaceDecl(xmlTextReaderPtr reader, HashTable * ht, ID id_list)
{
    ID id;
    EXT_ID ep;
    char * name = NULL;
    char * is_operator = NULL;
    char * is_assignment = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FinterfaceDecl"))
        return FALSE;

    name = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "name");
    is_operator = (char *) xmlTextReaderGetAttribute(reader,
                               BAD_CAST "is_operator");
    is_assignment = (char *) xmlTextReaderGetAttribute(reader,
                                 BAD_CAST "is_assignment");

    if (name != NULL) {
        id = find_ident_head(find_symbol(name), id_list);
        free(name);
    } else {
        assert(is_assignment != NULL); /* must be assignment */
        id = find_ident_head(find_symbol("="), id_list);
    }

    ep = PROC_EXT_ID(id);
    EXT_PROC_INTERFACE_INFO(ep) =
        XMALLOC(struct interface_info *, sizeof(struct interface_info));
    EXT_IS_BLANK_NAME(ep) = FALSE;
    EXT_PROC_CLASS(ep) = EP_INTERFACE;
    EXT_PROC_INTERFACE_CLASS(ep) = INTF_DECL;
    
    if (is_operator != NULL) {
        EXT_PROC_INTERFACE_CLASS(ep) = INTF_OPERATOR;
        free(is_operator);
    }

    if (is_assignment != NULL) {
        EXT_PROC_INTERFACE_CLASS(ep) = INTF_ASSINGMENT;
        free(is_assignment);
    }

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    while (!xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT)) {
        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                         "FmoduleProcedureDecl")) {
            if (!input_FmoduleProcedureDecl(reader, ht, ep))
                return FALSE;
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                                "FfunctionDecl")) {
            if (!input_FfunctionDecl(reader, ht, ep, id_list))
                return FALSE;
        }
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FinterfaceDecl"))
        return FALSE;

    return TRUE;
}

/**
 * input <interfaceDecls> node
 */
static int
input_interfaceDecls(xmlTextReaderPtr reader, HashTable * ht,
                     struct module * mod)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "interfaceDecls"))
        return FALSE;

    while (!xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT)) {
        if (!input_FinterfaceDecl(reader, ht, mod->head))
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "interfaceDecls"))
        return FALSE;

    return TRUE;
}

/**
 * input <OmniFortranModule> node
 */
static int
input_module(xmlTextReaderPtr reader, struct module * mod)
{
    char * version;
    SYMBOL mod_name;
    HashTable ht;

    InitHashTable(&ht, HASH_STRING_KEYS);

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
        "OmniFortranModule"))
        return FALSE;

    version = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "version");
    if (version == NULL || strcmp(version, "1.0") != 0) {
        if (version != NULL)
            free(version);
        return FALSE;
    }

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    /* <name> node */
    if (!input_name(reader, &mod_name))
        return FALSE;
    if (mod_name != mod->name) {
        fprintf(stderr, "module name '%s' in .xmod does'nt match to "
                "filename '%s.xmod'.", SYM_NAME(mod_name), SYM_NAME(mod->name));
        fflush(stderr);
    }

    /* <depends> node */
    if (!input_depends(reader, mod))
        return FALSE;

    /* <typeTable> node */
    if (!input_typeTable(reader, &ht))
        return FALSE;

    /* <identifiers> node */
    if (!input_identifiers(reader, &ht, mod))
        return FALSE;

    /* <interfaceDecls> node */
    if (!input_interfaceDecls(reader, &ht, mod))
        return FALSE;

    if(xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,"aux_info")){
	while(1){
	    if (!xmlTextReaderRead(reader)) return FALSE;
	    if(xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT,"aux_info"))
		break;
	}
	if (!xmlTextReaderRead(reader)) return FALSE;
	if (!xmlSkipWhiteSpace(reader)) return FALSE;
    }

    if (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT,
        "OmniFortranModule"))
        return FALSE;

    free(version);
    return TRUE;
}

/**
 * input module from .xmod file
 */
int
input_module_file(const SYMBOL mod_name, struct module **pmod)
{
    int ret;
    char filename[FILE_NAME_LEN];
    const char * filepath;
    xmlTextReaderPtr reader;

    bzero(filename, sizeof(filename));
    strcpy(filename, SYM_NAME(mod_name));
    strcat(filename, ".xmod");

    filepath = search_include_path(filename);

    reader = xmlNewTextReaderFilename(filepath);
    if (reader == NULL)
        return FALSE;

    *pmod = XMALLOC(struct module *, sizeof(struct module));
    (*pmod)->name = mod_name;

    ret = input_module(reader, *pmod);

    xmlTextReaderClose(reader);

    return ret;
}
