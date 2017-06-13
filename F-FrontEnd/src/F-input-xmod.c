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
    char * parent_type_id;
} * TYPE_ENTRY;

static int input_FfunctionDecl(xmlTextReaderPtr, HashTable *, EXT_ID, ID);
static int input_value(xmlTextReaderPtr, HashTable *, expv *);
static int input_varRef(xmlTextReaderPtr, HashTable *, expv *);
static int input_expv(xmlTextReaderPtr, HashTable *, expv *);

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
        fprintf(stderr, "On line %d, expected node type: %d, name: %s, "
                        "but was type: %d, name: %s\n",
                        xmlTextReaderGetParserLineNumber(reader),
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
        } else if (strcmp(typeId, "Fvoid") == 0) {
            TYPE_BASIC_TYPE(tp) = TYPE_VOID;
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

static void
setReturnType(HashTable * ht, TYPE_DESC ftp, const char * rtid)
{
    if (strcmp(rtid, "Fvoid") == 0) {
        TYPE_BASIC_TYPE(ftp) = TYPE_SUBR;
        FUNCTION_TYPE_RETURN_TYPE(ftp) = type_VOID;

    } else if (strncmp(rtid, "V", 1) == 0) {
        TYPE_DESC tp = getTypeDesc(ht, rtid);
        TYPE_BASIC_TYPE(ftp) = TYPE_GENERIC;
        FUNCTION_TYPE_SET_GENERIC(ftp);
        FUNCTION_TYPE_RETURN_TYPE(ftp) = tp;

        if (tp && TYPE_REF(tp) && TYPE_BASIC_TYPE(TYPE_REF(tp)) == TYPE_VOID) {
            TYPE_BASIC_TYPE(ftp) = TYPE_SUBR;
        } else {
            TYPE_BASIC_TYPE(ftp) = TYPE_FUNCTION;
        }

    } else {
        TYPE_BASIC_TYPE(ftp) = TYPE_FUNCTION;
        FUNCTION_TYPE_RETURN_TYPE(ftp) = getTypeDesc(ht, rtid);
    }
}

const char *
get_tagname_from_code(enum expr_code c)
{
    switch(c) {
    case PLUS_EXPR:
        return "plusExpr";
    case MINUS_EXPR:
        return "minusExpr";
    case UNARY_MINUS_EXPR:
        return "unaryMinusExpr";
    case MUL_EXPR:
        return "mulExpr";
    case DIV_EXPR:
        return "divExpr";
    case POWER_EXPR:
        return "FpowerExpr";
    case LOG_EQ_EXPR:
        return "logEQExpr";
    case LOG_NEQ_EXPR:
        return "logNEQExpr";
    case LOG_GE_EXPR:
        return "logGEExpr";
    case LOG_GT_EXPR:
        return "logGTExpr";
    case LOG_LE_EXPR:
        return "logLEExpr";
    case LOG_LT_EXPR:
        return "logLTExpr";
    case LOG_AND_EXPR:
        return "logAndExpr";
    case LOG_OR_EXPR:
        return "logOrExpr";
    case LOG_NOT_EXPR:
        return "logNotExpr";
    case F_EQV_EXPR:
        return "logEQVExpr";
    case F_NEQV_EXPR:
        return "logNEQVExpr";
    case F_CONCAT_EXPR:
        return "FconcatExpr";
    case F95_ARRAY_CONSTRUCTOR:
    case F03_TYPED_ARRAY_CONSTRUCTOR:
        return "FarrayConstructor";
    case F95_STRUCT_CONSTRUCTOR:
        return "FstructConstructor";
    default:
        return NULL;
    }
}

static int
input_name_as_string(xmlTextReaderPtr reader, char ** name)
{
    const char * str;
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name"))
       return FALSE;

    if (!xmlMatchNodeType(reader, XML_READER_TYPE_TEXT))
       return FALSE;

    str = (const char*) xmlTextReaderConstValue(reader);
    if (str == NULL)
        return FALSE;

    *name = strdup(str);

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name"))
        return FALSE;

    return TRUE;
}

/**
 * input <name> node
 */
static int
input_name(xmlTextReaderPtr reader, SYMBOL * s)
{
    char * name = NULL;

    if (!input_name_as_string(reader, &name))
       return FALSE;

    *s = find_symbol(name);

    free(name);

    return TRUE;
}

/**
 * input type and attribute at node
 */
static int
input_type_and_attr(xmlTextReaderPtr reader, HashTable * ht, char ** retTypeId,
                    TYPE_DESC * tp)
{
    char * str;
    char * typeId;

    typeId = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "type");
    if (typeId == NULL)
        return FALSE;

    *tp = getTypeDesc(ht, typeId);

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

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_volatile");
    if (str != NULL) {
        TYPE_SET_VOLATILE(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "extends");
    if (str != NULL) {
        TYPE_DESC parent_type = getTypeDesc(ht, str);
        if (parent_type != NULL) {
            TYPE_ENTRY tep = NULL;
            HashEntry * e;

            // TO BE FIXED lately
            TYPE_PARENT(*tp) = new_ident_desc(NULL);
            TYPE_PARENT_TYPE(*tp) = parent_type;
            e = FindHashEntry(ht, typeId);
            tep = GetHashValue(e);
            tep->parent_type_id = str;
        } else {
            // Error, but skip
            free(str);
        }
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "pass");
    if (str != NULL) {
        FUNCTION_TYPE_HAS_PASS_ARG(*tp) = TRUE;
        if (strcmp("pass", str) == 0) {
            FUNCTION_TYPE_HAS_PASS_ARG(*tp) = TRUE;
        } else if (strcmp("nopass", str) == 0) {
            FUNCTION_TYPE_HAS_PASS_ARG(*tp) = TRUE;
        } else {
            /* Unexpected */
            return FALSE;
        }
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "pass_arg_name");
    if (str != NULL) {
        ID pass_arg;
        pass_arg = new_ident_desc(find_symbol(str));
        FUNCTION_TYPE_PASS_ARG(*tp) = pass_arg;
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_class");
    if (str != NULL) {
        TYPE_SET_CLASS(*tp);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_procedure");
    if (str != NULL) {
        TYPE_SET_PROCEDURE(*tp);
        free(str);
    }

    if (retTypeId != NULL)
        *retTypeId = typeId;    /* return typeId */
    else
        free(typeId);

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
 * input FintConstant node
 */
static int
input_FintConstant(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    char * typeId = NULL;
    const char * value = NULL;
    TYPE_DESC tp = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FintConstant"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    value = (const char*) xmlTextReaderConstValue(reader);
    if (value == NULL)
        return FALSE;

    // FIXME: do we need this?
    // to be solved
    if (TYPE_BASIC_TYPE(tp) != TYPE_INT) {
        TYPE_BASIC_TYPE(tp) = TYPE_INT;
    }
    *v = expv_int_term(INT_CONSTANT, tp, atoll(value));

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FintConstant"))
        return FALSE;

    return TRUE;
}

/**
 * input FrealConstant node
 */
static int
input_FrealConstant(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    char * typeId = NULL;
    const char * value = NULL;
    TYPE_DESC tp = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FrealConstant"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    value = (const char*) xmlTextReaderConstValue(reader);
    if (value == NULL)
        return FALSE;

    /* FIXME: handling of double precision value */
    // to be solved
    if (TYPE_BASIC_TYPE(tp) != TYPE_REAL) {
        TYPE_BASIC_TYPE(tp) = TYPE_REAL;
    }
    *v = expv_float_term(FLOAT_CONSTANT, tp, atof(value), strdup(value));

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FrealConstant"))
        return FALSE;

    return TRUE;
}

/**
 * input FlogicalConstant node
 */
static int
input_FlogicalConstant(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    char * typeId = NULL;
    const char * value = NULL;
    TYPE_DESC tp = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FlogicalConstant"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    value = (const char*) xmlTextReaderConstValue(reader);
    if (value == NULL)
        return FALSE;

    *v = expv_int_term(INT_CONSTANT, tp,
             strcmp(value, ".TRUE.") == 0 ? TRUE : FALSE);

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FlogicalConstant"))
        return FALSE;

    return TRUE;
}

/**
 * input FcharacterConstant node
 */
static int
input_FcharacterConstant(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    char * typeId = NULL;
    const char * value = NULL;
    TYPE_DESC tp = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FcharacterConstant"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (xmlTextReaderRead(reader) != 1) return FALSE;

    value = (const char*) xmlTextReaderConstValue(reader);

    // FIXME: do we need this?
    // to be solved
    if (TYPE_BASIC_TYPE(tp) != TYPE_CHAR) {
        TYPE_BASIC_TYPE(tp) = TYPE_CHAR;
        TYPE_CHAR_LEN(tp) = 1;
    }

    if (value){
      *v = expv_str_term(STRING_CONSTANT, tp, (char *)value);
      if (xmlTextReaderRead(reader) != 1) return FALSE;
    }
    else {
      *v = expv_str_term(STRING_CONSTANT, tp, "");
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FcharacterConstant"))
        return FALSE;

    return TRUE;
}

/**
 * input FcomplexConstant node
 */
static int
input_FcomplexConstant(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    char * typeId = NULL;
    TYPE_DESC tp = NULL;
    expv i = NULL;
    expv r = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FcomplexConstant"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_expv(reader, ht, &i))
        return FALSE;

    if (!input_expv(reader, ht, &r))
        return FALSE;

    *v = list2(COMPLEX_CONSTANT, i, r);
    EXPV_TYPE(*v) = tp;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FcomplexConstant"))
        return FALSE;

    return TRUE;
}

/**
 * input Var node
 */
static int
input_Var(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    char * typeId = NULL;
    const char * name = NULL;
    TYPE_DESC tp = NULL;
    SYMBOL s;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "Var"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    name = (const char*) xmlTextReaderConstValue(reader);
    if (name == NULL)
        return FALSE;

    s = find_symbol(name);
    *v = expv_sym_term(F_VAR, tp, s);

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "Var"))
        return FALSE;

    return TRUE;
}

/**
 * input unary expression node
 */
static int
input_unaryExpr(xmlTextReaderPtr reader, HashTable * ht, enum expr_code c,
    expv * v)
{
    expv operand;
    TYPE_DESC tp = NULL;
    const char * tag = get_tagname_from_code(c);
    char * typeId = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, tag))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_expv(reader, ht, &operand))
        return FALSE;

    *v = expv_cons(c, tp, operand, NULL);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, tag))
        return FALSE;

    return TRUE;
}

/**
 * input binary expression node
 */
static int
input_binaryExpr(xmlTextReaderPtr reader, HashTable * ht, enum expr_code c,
    expv * v)
{
    expv lt, rt;
    TYPE_DESC tp = NULL;
    const char * tag = get_tagname_from_code(c);
    char * typeId = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, tag))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_expv(reader, ht, &lt))
        return FALSE;

    if (!input_expv(reader, ht, &rt))
        return FALSE;

    *v = expv_cons(c, tp, lt, rt);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, tag))
        return FALSE;

    return TRUE;
}

/**
 * input multiple(zero or more) expression node
 */
static int
input_multipleExpr(xmlTextReaderPtr reader, HashTable * ht, enum expr_code c,
    expv * v)
{
    expv operand;
    expv list;
    TYPE_DESC tp = NULL;
    const char * tag = get_tagname_from_code(c);
    char * typeId = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, tag))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    list = list0(LIST);

    while (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, tag)) {
        if (!input_expv(reader, ht, &operand))
            return FALSE;
        list_put_last(list, operand);
    }

    *v = expv_cons(c, tp, list, NULL);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, tag))
        return FALSE;

    return TRUE;
}

static int
input_typeParamValues(xmlTextReaderPtr reader, HashTable * ht, expv * typeParamValues)
{
    expv value;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "typeParamValues"))
        return FALSE;

    *typeParamValues = list0(LIST);

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    while (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "typeParamValues")) {
        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "namedValue")) {
            char * name = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "name");
            if (name == NULL)
                return FALSE;

            if (!xmlSkipWhiteSpace(reader))
                return FALSE;

            if (!input_expv(reader, ht, &value))
                return FALSE;

            EXPV_KWOPT_NAME(value) = strdup(name);
            free(name);

            if (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "namedValue"))
                return FALSE;

            if (!xmlSkipWhiteSpace(reader))
                return FALSE;

        } else {

            if (!input_expv(reader, ht, &value))
                return FALSE;

            if (!xmlSkipWhiteSpace(reader))
                return FALSE;
        }
        list_put_last(*typeParamValues, value);
    }

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    return TRUE;
}


/**
 * input multiple(zero or more) expression node
 */
static int
input_FstructConstructor(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv operand;
    expv typeParamValues = NULL;
    expv components;
    TYPE_DESC tp = NULL;
    char * typeId = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FstructConstructor"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "typeParamValues")) {
        if (!input_typeParamValues(reader, ht, &typeParamValues))
            return FALSE;
    }

    components = list0(LIST);

    while (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "FstructConstructor")) {
        if (!input_expv(reader, ht, &operand))
            return FALSE;
        list_put_last(components, operand);
    }

    *v = expv_cons(F95_STRUCT_CONSTRUCTOR, tp, typeParamValues, components);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FstructConstructor"))
        return FALSE;

    return TRUE;
}

/**
 * input user binary expression node
 */
static int
input_userBinaryExpr(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv lt, rt;
    TYPE_DESC tp = NULL;
    char * typeId = NULL;
    char * name = NULL;
    SYMBOL s;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "userBinaryExpr"))
        return FALSE;

    name = (char *)xmlTextReaderGetAttribute(reader, BAD_CAST "name");
    if (name == NULL)
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_expv(reader, ht, &lt))
        return FALSE;

    if (!input_expv(reader, ht, &rt))
        return FALSE;

    s = find_symbol(name);
    *v = list3(F95_USER_DEFINED_BINARY_EXPR,
               expv_sym_term(IDENT, NULL, s), lt, rt);
    EXPV_TYPE(*v) = tp;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "userBinaryExpr"))
        return FALSE;

    return TRUE;
}

/**
 * input user unary expression node
 */
static int
input_userUnaryExpr(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv operand;
    TYPE_DESC tp = NULL;
    char * typeId = NULL;
    char * name = NULL;
    SYMBOL s;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "userUnaryExpr"))
        return FALSE;

    name = (char *)xmlTextReaderGetAttribute(reader, BAD_CAST "name");
    if (name == NULL)
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_expv(reader, ht, &operand))
        return FALSE;

    s = find_symbol(name);
    *v = list2(F95_USER_DEFINED_UNARY_EXPR,
               expv_sym_term(IDENT, NULL, s), operand);
    EXPV_TYPE(*v) = tp;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "userUnaryExpr"))
        return FALSE;

    return TRUE;
}

/**
 * input FmemberRef node
 */
static int
input_FmemberRef(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv v1, v2;
    TYPE_DESC tp = NULL;
    char * typeId = NULL;
    const char * member = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FmemberRef"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    member = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "member");
    if (member == NULL)
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_varRef(reader, ht, &v1))
        return FALSE;

    v2 = expv_sym_term(IDENT, NULL, find_symbol(member));
    *v = expv_cons(F95_MEMBER_REF, tp, v1, v2);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FmemberRef"))
        return FALSE;

    return TRUE;
}

/**
 * input functionCall node
 */
static int
input_functionCall(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    char * name;
    char * typeId = NULL;
    SYMBOL s;
    ID id;
    TYPE_DESC tp = NULL;
    expv arg;
    expv args;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "functionCall"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_name_as_string(reader, &name))
        return FALSE;

    args = list0(LIST);

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "arguments")) {

        if (!xmlSkipWhiteSpace(reader))
            return FALSE;

        while (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "arguments")) {
            if (!input_expv(reader, ht, &arg))
                return FALSE;
            list_put_last(args, arg);
        }

        if (!xmlSkipWhiteSpace(reader))
            return FALSE;
    }

    s = find_symbol(name);
    SYM_TYPE(s) = TYPE_IS_INTRINSIC(tp) ? S_INTR : S_IDENT;
    id = new_ident_desc(s);
    PROC_EXT_ID(id) = new_external_id_for_external_decl(s, tp);

    if (TYPE_IS_INTRINSIC(tp)) {
        *v = compile_intrinsic_call(id, args);
    } else {
        ID_ADDR(id) = expv_sym_term(IDENT, NULL, s);
        *v = list3(FUNCTION_CALL, ID_ADDR(id), args, expv_any_term(F_EXTFUNC, id));
        EXPV_TYPE(*v) = getTypeDesc(ht, typeId);
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "functionCall"))
        return FALSE;

    return TRUE;
}

/**
 * input <kind> node
 */
static int
input_kind(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC tp)
{
    expv v;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "kind"))
        return TRUE;

    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (!input_expv(reader, ht, &v))
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
input_len(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC tp)
{
    expv v = NULL;
    char * is_assumed_size;
    char * is_assumed_shape;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "len"))
        return TRUE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    is_assumed_size = (char *) xmlTextReaderGetAttribute(reader,
                                   BAD_CAST "is_assumed_size");

    is_assumed_shape = (char *) xmlTextReaderGetAttribute(reader,
                                    BAD_CAST "is_assumed_shape");


    if (xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "len")) {
        /* if <len> tag is empty, size is unfixed */
        TYPE_CHAR_LEN(tp) = CHAR_LEN_UNFIXED;
    } else {
        if (!input_expv(reader, ht, &v))
            return FALSE;

        if (v != NULL)
            TYPE_LENG(tp) = v;
    }

    if (is_assumed_size != NULL) {
        TYPE_CHAR_LEN(tp) = CHAR_LEN_UNFIXED;
        free(is_assumed_size);
    } else if (is_assumed_shape != NULL) {
        TYPE_CHAR_LEN(tp) = CHAR_LEN_ALLOCATABLE;
        free(is_assumed_shape);
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "len"))
        return FALSE;

    return TRUE;
}

/**
 * input <lowerBound> node
 */
static int
input_lowerBound(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "lowerBound"))
        return FALSE;

    if (!input_expv(reader, ht, v))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "lowerBound"))
        return FALSE;

    return TRUE;
}

/**
 * input <upperBound> node
 */
static int
input_upperBound(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "upperBound"))
        return FALSE;

    if (!input_expv(reader, ht, v))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "upperBound"))
        return FALSE;

    return TRUE;
}

/**
 * input <step> node
 */
static int
input_step(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "step"))
        return FALSE;

    if (!input_expv(reader, ht, v))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "step"))
        return FALSE;

    return TRUE;
}

/**
 * input <indexRange> node
 */
static int
input_indexRange(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC tp)
{
    TYPE_DESC bottom, base;
    expv v;
    char * is_assumed_size;
    char * is_assumed_shape;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "indexRange"))
        return FALSE;

    bottom = tp;
    base = new_type_desc();
    *base = *bottom;
    TYPE_BASIC_TYPE(bottom) = TYPE_ARRAY;
    TYPE_REF(bottom) = base;
    TYPE_N_DIM(bottom) = TYPE_N_DIM(base)+1;
    
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
        if (input_lowerBound(reader, ht, &v))
            TYPE_DIM_LOWER(bottom) = v;
        else
            return FALSE;
    }

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "upperBound")) {
        if (input_upperBound(reader, ht, &v))
            TYPE_DIM_UPPER(bottom) = v;
        else
            return FALSE;
    }

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "step")) {
        if (input_step(reader, ht, &v))
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
input_indexRange_coShape(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC tp)
{
    char * is_assumed_size;
    char * is_assumed_shape;
    ARRAY_ASSUME_KIND assumeKind;
    expv cobound = NULL;
    expv lower = NULL;
    expv upper = NULL;
    expv step = NULL;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "indexRange"))
        return FALSE;

    is_assumed_size = (char *) xmlTextReaderGetAttribute(reader,
                                   BAD_CAST "is_assumed_size");

    is_assumed_shape = (char *) xmlTextReaderGetAttribute(reader,
                                    BAD_CAST "is_assumed_shape");

    if (is_assumed_size != NULL) {
        assumeKind = ASSUMED_SIZE;
        free(is_assumed_size);
    } else if (is_assumed_shape != NULL) {
        assumeKind = ASSUMED_SHAPE;
        free(is_assumed_shape);
    } else {
        assumeKind = ASSUMED_NONE;
    }

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "lowerBound"))
        if (!input_lowerBound(reader, ht, &lower))
            return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "upperBound"))
        if (!input_upperBound(reader, ht, &upper))
            return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "step"))
        if (!input_step(reader, ht, &step))
            return FALSE;

    if(upper == NULL && assumeKind == ASSUMED_SIZE){
        upper = expv_any_term(F_ASTERISK, NULL);
    }

    cobound = list3(LIST, lower, upper, step);
    tp->codims->cobound_list = list_put_last(tp->codims->cobound_list, cobound);
    tp->codims->corank++;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "indexRange"))
        return FALSE;

    return TRUE;
}

/**
 * input <coShape> node
 */
static int
input_coShape(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC tp)
{
    codims_desc * codims;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "coShape"))
        return FALSE;

    codims = XMALLOC(codims_desc *, sizeof(*codims));
    tp->codims = codims;
    codims->cobound_list = EMPTY_LIST;

    while (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "indexRange")) {
        if (!input_indexRange_coShape(reader, ht, tp))
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "coShape"))
        return FALSE;

    return TRUE;
}

/**
 * input indexRange or arrayIndex in FarrayRef, FcoArrayRef node
 */
static int
input_arraySpec(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv lower = NULL;
    expv upper = NULL;
    expv step = NULL;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "arrayIndex")) {
        if (!xmlSkipWhiteSpace(reader))
            return FALSE;

        if (!input_expv(reader, ht, v))
            return FALSE;

        if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "arrayIndex"))
            return FALSE;

    } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "indexRange")) {
        if (!xmlSkipWhiteSpace(reader))
            return FALSE;

        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "lowerBound"))
            if (!input_lowerBound(reader, ht, &lower))
                return FALSE;

        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "upperBound"))
            if (!input_upperBound(reader, ht, &upper))
                return FALSE;

        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "step"))
            if (!input_step(reader, ht, &step))
                return FALSE;

        *v = list3(F_INDEX_RANGE, lower, upper, step);

        if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "indexRange"))
            return FALSE;
    }

    return TRUE;
}

/**
 * input FarrayRef node
 */
static int
input_FarrayRef(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv v1, v2, v3;
    TYPE_DESC tp = NULL;
    char * typeId = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FarrayRef"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_varRef(reader, ht, &v1))
        return FALSE;

    v2 = list0(LIST);

    while (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "FarrayRef")) {
        if (!input_arraySpec(reader, ht, &v3))
            return FALSE;
        list_put_last(v2, v3);
    }

    *v = expv_cons(ARRAY_REF, tp, v1, v2);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FarrayRef"))
        return FALSE;

    return TRUE;
}

/**
 * input FcoArrayRef node
 */
static int
input_FcoArrayRef(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv v1, v2, v3;
    TYPE_DESC tp = NULL;
    char * typeId = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FcoArrayRef"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_varRef(reader, ht, &v1))
        return FALSE;

    v2 = list0(LIST);

    while (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "FcoArrayRef")) {
        if (!input_arraySpec(reader, ht, &v3))
            return FALSE;
        list_put_last(v2, v3);
    }

    *v = expv_cons(XMP_COARRAY_REF, tp, v1, v2);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FcoArrayRef"))
        return FALSE;

    return TRUE;
}

/**
 * input FcharacterRef node
 */
static int
input_FcharacterRef(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv v1, v2;
    TYPE_DESC tp = NULL;
    char * typeId = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FcharacterRef"))
        return FALSE;

    if (!input_type_and_attr(reader, ht, &typeId, &tp))
        return FALSE;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!input_varRef(reader, ht, &v1))
        return FALSE;

    v2 = NULL;

    if (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "FcharacterRef")) {
        if (!input_arraySpec(reader, ht, &v2))
            return FALSE;
    }

    *v = expv_cons(F_SUBSTR_REF, tp, v1, v2);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FcharacterRef"))
        return FALSE;

    return TRUE;
}

/**
 * input varRef node
 */
static int
input_varRef(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    int ret = FALSE;
    const char * name;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "varRef"))
        return FALSE;

    name = (const char *)xmlTextReaderConstName(reader);
    if (strcmp(name, "Var") == 0)
        ret = input_Var(reader, ht, v);
    else if (strcmp(name, "FarrayRef") == 0)
        ret = input_FarrayRef(reader, ht, v);
    else if (strcmp(name, "FcharacterRef") == 0)
        ret = input_FcharacterRef(reader, ht, v);
    else if (strcmp(name, "FmemberRef") == 0)
        ret = input_FmemberRef(reader, ht, v);
    else if (strcmp(name, "FcoArrayRef") == 0)
        ret = input_FcoArrayRef(reader, ht, v);

    if (!ret)
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "varRef"))
        return FALSE;

    return TRUE;
}

/**
 * input FdoLoop node
 */
static int
input_FdoLoop(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    expv v1, v2;
    expv var, lower, upper, step;
    expv value;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "FdoLoop"))
        return FALSE;

    if (!input_Var(reader, ht, &var))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "indexRange"))
        return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "lowerBound"))
        if (!input_lowerBound(reader, ht, &lower))
            return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "upperBound"))
        if (!input_upperBound(reader, ht, &upper))
            return FALSE;

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "step"))
        if (!input_step(reader, ht, &step))
            return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "indexRange"))
        return FALSE;

    v1 = list4(LIST, var, lower, upper, step);
    v2 = list0(LIST);

    while (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "FdoLoop")) {
        if (!input_value(reader, ht, &value))
            return FALSE;
        list_put_last(v2, value);
    }

    *v = expv_cons(F_IMPLIED_DO, NULL, v1, v2);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FdoLoop"))
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
    if (ref != NULL) {
        TYPE_REF(tp) = getTypeDesc(ht, ref);
        TYPE_BASIC_TYPE(tp) = TYPE_BASIC_TYPE(TYPE_REF(tp));
        shrink_type(tp);

        if (IS_CHAR(tp))  {
            TYPE_CHAR_LEN(tp) = TYPE_CHAR_LEN(TYPE_REF(tp));
        }

    } else {
        TYPE_REF(tp) = NULL;
        if (TYPE_IS_PROCEDURE(tp)) {
            TYPE_BASIC_TYPE(tp) = TYPE_FUNCTION;
        } else if (TYPE_IS_CLASS(tp)) {
            TYPE_BASIC_TYPE(tp) = TYPE_STRUCT;
        }
    }

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (isEmpty)
        return TRUE;	/* case FbasicType node has no child */

    /* <kind> */
    if (!input_kind(reader, ht, TYPE_REF(tp)))    /* kind should be set to ref */
        return FALSE;

    /* <len> */
    if (!input_len(reader, ht, tp))
        return FALSE;

    if (IS_CHAR(tp) && TYPE_LENG(tp))  {
        if (EXPR_CODE(TYPE_LENG(tp)) == INT_CONSTANT) {
            TYPE_CHAR_LEN(tp) = EXPV_INT_VALUE(TYPE_LENG(tp));
        } else {
            /*
             * don't use as a character basictype "Fcharacter"
             */
            TYPE_CHAR_LEN(tp) = 0;
        }
    }

    /* <indexRange> */
    while (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "indexRange")) {
        if (!input_indexRange(reader, ht, tp))
            return FALSE;
    }

    /* <coShape> */
    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "coShape"))
        if (!input_coShape(reader, ht, tp))
            return FALSE;

    /* <typeParamValues> */
    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "typeParamValues")) {
        expv typeParamValues;
        if (!input_typeParamValues(reader, ht, &typeParamValues))
            return FALSE;

        TYPE_TYPE_PARAM_VALUES(tp) = typeParamValues;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FbasicType"))
        return FALSE;

    if (typeId != NULL)
        free(typeId);

    /*
     * Remove a character basic type which is genereted from 'ref="Fcharacter"'
     */
    if (IS_CHAR(tp))  {
        if (TYPE_REF(TYPE_REF(tp)) == NULL &&
            TYPE_CHAR_LEN(TYPE_REF(tp)) == 1) {
            TYPE_REF(tp) = NULL;
        }
    }

    return TRUE;
}

/**
 * input <name> node with type attribute appears in <params>
 */
static int
input_param(xmlTextReaderPtr reader, HashTable * ht, EXT_ID ep, TYPE_DESC ftp)
{
    TYPE_DESC tp = NULL;
    SYMBOL s;
    expv v;
    ID ftp_arg;

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

    ftp_arg = new_ident_desc(s);
    ID_TYPE(ftp_arg) = tp;

    if (FUNCTION_TYPE_ARGS(ftp) == NULL) {
        FUNCTION_TYPE_ARGS(ftp) = ftp_arg;
    } else {
        ID last = NULL;
        last = FUNCTION_TYPE_ARGS(ftp);
        while (ID_NEXT(last) != NULL) {
            last = ID_NEXT(last);
        }
        ID_NEXT(last) = ftp_arg;
    }

    return TRUE;
}

/**
 * input <FfunctionType> node
 */
static int
input_FfunctionType(xmlTextReaderPtr reader, HashTable * ht)
{
    TYPE_ENTRY tep;
    TYPE_DESC ftp = NULL;
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
    ftp = tep->tp;

    EXT_PROC_ARGS(tep->ep) = EMPTY_LIST;
    EXT_IS_DEFINED(tep->ep) = TRUE;
    EXT_IS_OFMODULE(tep->ep) = FALSE;
    EXT_TAG(tep->ep) = STG_EXT;

    setReturnType(ht, ftp,
                  (char *) xmlTextReaderGetAttribute(reader,
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
        TYPE_SET_RECURSIVE(ftp);
        free(attr);
    }

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_pure");
    if (attr != NULL) {
        TYPE_SET_PURE(ftp);
        free(attr);
    }

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_elemental");
    if (attr != NULL) {
        TYPE_SET_ELEMENTAL(ftp);
        free(attr);
    }

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_module");
    if (attr != NULL) {
        TYPE_SET_MODULE(ftp);
        free(attr);
    }

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_external");
    if (attr != NULL) {
        TYPE_SET_EXTERNAL(ftp);
        free(attr);
    }

    attr = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_defined");
    if (attr != NULL) {
        FUNCTION_TYPE_SET_DEFINED(ftp);
        free(attr);
    }


    if (!xmlSkipWhiteSpace(reader)) 
        return FALSE;

    if (isEmpty)
        return TRUE;	/* case FfunctionType node has no child */

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "params"))
        return FALSE;

    FUNCTION_TYPE_HAS_EXPLICIT_ARGS(ftp) = TRUE;

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </params> */
            break;

        if (!input_param(reader, ht, tep->ep, ftp))
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
    expv v;
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

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "value")) {
        if (!input_value(reader, ht, &v))
            return FALSE;
        VAR_INIT_VALUE(id) = v;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "id"))
        return FALSE;

    return TRUE;
}


/**
 * input <typeParam> node
 */
static int
input_typeParam(xmlTextReaderPtr reader, HashTable * ht, ID * id)
{
    TYPE_DESC tp;
    SYMBOL s;
    char * str = NULL;
    char * name = NULL;
    *id = NULL;

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "type");
    if (str == NULL)
        return FALSE;

    tp = getTypeDesc(ht, str);
    free(str);

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "attr");
    if (str == NULL)
        return FALSE;

    if (strcmp(str, "kind") == 0)
        TYPE_SET_KIND(tp);
    else if (strcmp(str, "length") == 0)
        TYPE_SET_LEN(tp);
    free(str);

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "name")) {
        return FALSE;
    }

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    name = (char *)xmlTextReaderConstValue(reader);
    if (name != NULL) {
        name = strdup(name);
    }

    s = find_symbol(name);
    SYM_TYPE(s) = S_IDENT;
    *id = new_ident_desc(s);
    ID_TYPE(*id) = tp;
    free(name);

    if (!xmlSkipWhiteSpace(reader)) {
        free(*id);
        *id = NULL;
        return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name")) {
        free(*id);
        *id = NULL;
        return FALSE;
    }

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "value")) {
        expv v;
        if (!input_value(reader, ht, &v)) {
            free(*id);
            *id = NULL;
            return FALSE;
        }
        VAR_INIT_VALUE(*id) = v;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "typeParam")) {
        free(*id);
        *id = NULL;
        return FALSE;
    }

    return TRUE;
}

/**
 * input <typeParams> node
 */
static int
input_typeParams(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC struct_tp)
{
    ID id, last = NULL;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "typeParams"))
        return FALSE;

    TYPE_TYPE_PARAMS(struct_tp) = NULL;

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </typeParams> */
            break;

        if (!input_typeParam(reader, ht, &id))
            return FALSE;

        ID_LINK_ADD(id, TYPE_TYPE_PARAMS(struct_tp), last);
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "typeParams"))
        return FALSE;

    return TRUE;
}


static int
input_typeBoundProcedure(xmlTextReaderPtr reader, HashTable * ht, ID * id)
{
    char * name;
    char * typeId;
    char * str;
    ID binding = NULL;
    ID pass_arg = NULL;
    TYPE_ENTRY tep;
    TYPE_DESC ftp;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                       "typeBoundProcedure"))
        return FALSE;

    typeId = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "type");
    if (typeId == NULL)
        return FALSE;

    tep = getTypeEntry(ht, typeId);
    ftp = tep->tp;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name"))
        return FALSE;

    name = (char *)xmlTextReaderConstValue(reader);
    if (name != NULL) {
        name = strdup(name);
    }
    if (!xmlSkipWhiteSpace(reader)) {
        return FALSE;
    }

    *id = new_ident_desc(find_symbol(name));
    ID_CLASS(*id) = CL_TYPE_BOUND_PROC;
    ID_TYPE(*id) = type_bound_procedure_type();
    TYPE_REF(ID_TYPE(*id)) = ftp;

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "pass");
    if (str != NULL) {
        if (strcmp("pass", str) == 0) {
            TBP_BINDING_ATTRS(*id) |= TYPE_BOUND_PROCEDURE_PASS;
        } else if (strcmp("nopass", str) == 0) {
            TBP_BINDING_ATTRS(*id) |= TYPE_BOUND_PROCEDURE_NOPASS;
        } else {
            /* Unexpected */
            return FALSE;
        }
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "pass_arg_name");
    if (str != NULL) {
        pass_arg = new_ident_desc(find_symbol(str));
        free(str);
    }
    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_non_overridable");
    if (str != NULL) {
        TBP_BINDING_ATTRS(*id) |= TYPE_BOUND_PROCEDURE_NON_OVERRIDABLE;
        free(str);
    }
    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_deferred");
    if (str != NULL) {
        TBP_BINDING_ATTRS(*id) |= TYPE_BOUND_PROCEDURE_DEFERRED;
        free(str);
    }
    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_public");
    if (str != NULL) {
        TYPE_SET_PUBLIC(*id);
        free(str);
    }
    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_private");
    if (str != NULL) {
        TYPE_SET_PRIVATE(*id);
        free(str);
    }

    TBP_PASS_ARG(*id) = pass_arg;
    TBP_BINDING(*id) = binding;
    TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(ID_TYPE(*id)) =
            TBP_BINDING_ATTRS(*id) & TYPE_BOUND_PROCEDURE_PASS;
    TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(ID_TYPE(*id)) = pass_arg;

    if (!xmlSkipWhiteSpace(reader))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "binding"))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name"))
        return FALSE;

    name = (char *)xmlTextReaderConstValue(reader);
    if (name != NULL) {
        name = strdup(name);
    }
    if (!xmlSkipWhiteSpace(reader)) {
        return FALSE;
    }

    binding = new_ident_desc(find_symbol(name));
    TBP_BINDING(*id) = binding;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name"))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "binding"))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT,
                       "typeBoundProcedure"))
        return FALSE;

    return TRUE;
}


static int
input_typeBoundGenericProcedure(xmlTextReaderPtr reader, HashTable * ht, ID *id)
{
    char * name = NULL;
    char * str;
    ID binding = NULL;
    ID pass_arg = NULL;
    ID last_ip = NULL;
    uint32_t binding_attr_flags = TYPE_BOUND_PROCEDURE_IS_GENERIC;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                       "typeBoundProcedure"))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name"))
        return FALSE;

    name = (char *)xmlTextReaderConstValue(reader);
    if (name != NULL) {
        name = strdup(name);
    }
    if (!xmlSkipWhiteSpace(reader)) {
        return FALSE;
    }

    *id = new_ident_desc(find_symbol(name));
    TBP_BINDING_ATTRS(*id) |= TYPE_BOUND_PROCEDURE_IS_GENERIC;

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_public");
    if (str != NULL) {
        TYPE_SET_PUBLIC(*id);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_private");
    if (str != NULL) {
        TYPE_SET_PRIVATE(*id);
        free(str);
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_operator");
    if (str != NULL) {
        binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_OPERATOR;
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_assignment");
    if (str != NULL) {
        binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_ASSIGNMENT;
    }

    str = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_defined_io");
    if (str != NULL) {
        if (strcmp("WRITE(FORMATTED)", str) == 0) {
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_WRITE;
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_FORMATTED;
            name = "_write_formatted";

        } else if (strcmp("WRITE(UNFORMATTED)", str) == 0) {
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_WRITE;
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_UNFORMATTED;
            name = "_write_unformatted";

        } else if (strcmp("READ(FORMATTED)", str) == 0) {
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_READ;
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_FORMATTED;
            name = "_read_formatted";

        } else if (strcmp("READ(UNFORMATTED)", str) == 0) {
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_READ;
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_UNFORMATTED;
            name = "_read_unformatted";
        } else {
            return FALSE;
        }
    }

    TBP_PASS_ARG(*id) = pass_arg;
    TBP_BINDING(*id) = NULL;
    TBP_BINDING_ATTRS(*id) = binding_attr_flags;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "binding"))
        return FALSE;

    while (TRUE) {
        if (xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name")) {
            name = (char *)xmlTextReaderConstValue(reader);
            if (!xmlSkipWhiteSpace(reader)) {
                return FALSE;
            }
        }

        if (name != NULL) {
            name = strdup(name);
        }

        binding = new_ident_desc(find_symbol(name));
        ID_LINK_ADD(binding, TBP_BINDING(*id), last_ip);

        if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name"))
            return FALSE;

        if (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "binding"))
            break;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "binding"))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT,
                       "typeBoundProcedure"))
        return FALSE;

    return TRUE;
}


static int
input_finalProcedure(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC stp)
{
    char * name = NULL;
    ID binding;
    ID mem;
    ID last_ip = NULL;
    ID id = NULL;
    SYMBOL sym = find_symbol(FINALIZER_PROCEDURE);

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                       "finalProcedure"))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name"))
        return FALSE;

    name = (char *)xmlTextReaderConstValue(reader);
    if (name != NULL) {
        name = strdup(name);
    }
    if (!xmlSkipWhiteSpace(reader)) {
        return FALSE;
    }

    id = find_struct_member(stp, sym);
    if (id == NULL) {
        ID last = NULL;
        id = new_ident_desc(sym);
        ID_LINK_ADD(id, TYPE_MEMBER_LIST(stp), last);
    }

    if (xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "name")) {
        name = (char *)xmlTextReaderConstValue(reader);
        if (!xmlSkipWhiteSpace(reader)) {
            return FALSE;
        }
    }

    if (name != NULL) {
        name = strdup(name);
    }

    binding = new_ident_desc(find_symbol(name));
    FOREACH_ID(mem, TBP_BINDING(id)) {
        last_ip = mem;
    }
    ID_LINK_ADD(binding, TBP_BINDING(id), last_ip);

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "name"))
        return FALSE;

    if (!xmlMatchNode(reader, XML_READER_TYPE_END_ELEMENT, "finalProcedure"))
        return FALSE;

    return TRUE;
}


static int
input_typeBoundProcedures(xmlTextReaderPtr reader, HashTable * ht, TYPE_DESC struct_tp)
{
    ID mem = NULL;
    ID last_ip = NULL;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "typeBoundProcedures"))
        return FALSE;

    FOREACH_MEMBER(mem, struct_tp) {
        last_ip = mem;
    }

    while (TRUE) {
        if (xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT))
            /* must be </typeParams> */
            break;

        if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                         "typeBoundProcedure")) {
            if (!input_typeBoundProcedure(reader, ht, &mem))
                return FALSE;
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                                "typeBoundGenericProcedure")) {
            if (!input_typeBoundGenericProcedure(reader, ht, &mem))
                return FALSE;
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                                "finalProcedure")) {
            if (!input_finalProcedure(reader, ht, struct_tp))
                return FALSE;
        }

        ID_LINK_ADD(mem, TYPE_MEMBER_LIST(struct_tp), last_ip);
    }


    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT,
                       "typeBoundProcedures"))
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

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "typeParams")) {
        if (!input_typeParams(reader, ht, tp))
            return FALSE;
    }

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

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "typeBoundProcedures")) {
        if (!input_typeBoundProcedures(reader, ht, tp))
            return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "FstructType"))
        return FALSE;

    return TRUE;
}

/**
 * input <value> node
 */
static int
input_value(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "value"))
        return FALSE;

    if (!input_expv(reader, ht, v))
        return FALSE;

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "value"))
        return FALSE;

    return TRUE;
}

/**
 * input expv node
 */
static int
input_expv(xmlTextReaderPtr reader, HashTable * ht, expv * v)
{
    const char * name;
    name = (const char *)xmlTextReaderConstName(reader);

    if (strcmp(name, "FintConstant") == 0)
        return input_FintConstant(reader, ht, v);
    if (strcmp(name, "FrealConstant") == 0)
        return input_FrealConstant(reader, ht, v);
    if (strcmp(name, "FcomplexConstant") == 0)
        return input_FcomplexConstant(reader, ht, v);
    if (strcmp(name, "FcharacterConstant") == 0)
        return input_FcharacterConstant(reader, ht, v);
    if (strcmp(name, "FlogicalConstant") == 0)
        return input_FlogicalConstant(reader, ht, v);
    if (strcmp(name, "FarrayConstructor") == 0)
        return input_multipleExpr(reader, ht, F95_ARRAY_CONSTRUCTOR, v);
    if (strcmp(name, "FstructConstructor") == 0)
        return input_FstructConstructor(reader, ht, v);
    if (strcmp(name, "Var") == 0)
        return input_Var(reader, ht, v);
    if (strcmp(name, "FarrayRef") == 0)
        return input_FarrayRef(reader, ht, v);
    if (strcmp(name, "FcharacterRef") == 0)
        return input_FcharacterRef(reader, ht, v);
    if (strcmp(name, "FmemberRef") == 0)
        return input_FmemberRef(reader, ht, v);
    if (strcmp(name, "FcoArrayRef") == 0)
        return input_FcoArrayRef(reader, ht, v);
    if (strcmp(name, "varRef") == 0)
        return input_varRef(reader, ht, v);
    if (strcmp(name, "functionCall") == 0)
        return input_functionCall(reader, ht, v);
    if (strcmp(name, "plusExpr") == 0)
        return input_binaryExpr(reader, ht, PLUS_EXPR, v);
    if (strcmp(name, "minusExpr") == 0)
        return input_binaryExpr(reader, ht, MINUS_EXPR, v);
    if (strcmp(name, "mulExpr") == 0)
        return input_binaryExpr(reader, ht, MUL_EXPR, v);
    if (strcmp(name, "divExpr") == 0)
        return input_binaryExpr(reader, ht, DIV_EXPR, v);
    if (strcmp(name, "FpowerExpr") == 0)
        return input_binaryExpr(reader, ht, POWER_EXPR, v);
    if (strcmp(name, "FconcatExpr") == 0)
        return input_binaryExpr(reader, ht, F_CONCAT_EXPR, v);
    if (strcmp(name, "logEQExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_EQ_EXPR, v);
    if (strcmp(name, "logNEQExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_NEQ_EXPR, v);
    if (strcmp(name, "logGEExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_GE_EXPR, v);
    if (strcmp(name, "logGTExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_GT_EXPR, v);
    if (strcmp(name, "logLEExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_LE_EXPR, v);
    if (strcmp(name, "logLTExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_LT_EXPR, v);
    if (strcmp(name, "logAndExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_AND_EXPR, v);
    if (strcmp(name, "logOrExpr") == 0)
        return input_binaryExpr(reader, ht, LOG_OR_EXPR, v);
    if (strcmp(name, "logEQVExpr") == 0)
        return input_binaryExpr(reader, ht, F_EQV_EXPR, v);
    if (strcmp(name, "logNEQVExpr") == 0)
        return input_binaryExpr(reader, ht, F_NEQV_EXPR, v);
    if (strcmp(name, "unaryMinusExpr") == 0)
        return input_unaryExpr(reader, ht, UNARY_MINUS_EXPR, v);
    if (strcmp(name, "logNotExpr") == 0)
        return input_unaryExpr(reader, ht, LOG_NOT_EXPR, v);
    if (strcmp(name, "userBinaryExpr") == 0)
        return input_userBinaryExpr(reader, ht, v);
    if (strcmp(name, "userUnaryExpr") == 0)
        return input_userUnaryExpr(reader, ht, v);
    if (strcmp(name, "FdoLoop") == 0)
        return input_FdoLoop(reader, ht, v);

    fprintf(stderr, "unknown node \"%s\".\n", name);

    return FALSE;
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

        if (!TYPE_IS_PROCEDURE(tp) &&
            (IS_PROCEDURE_TYPE(tp) || IS_GENERIC_TYPE(tp))) {
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
    id->use_assoc->module = mod;
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
        if ((TYPE_BASIC_TYPE(ID_TYPE(id)) == TYPE_FUNCTION ||
             TYPE_BASIC_TYPE(ID_TYPE(id)) == TYPE_SUBR ||
             TYPE_BASIC_TYPE(ID_TYPE(id)) == TYPE_GENERIC) &&
            !TYPE_IS_PROCEDURE(ID_TYPE(id))) {
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
            if (tep->ep != NULL && EXT_PROC_IS_ENTRY(tep->ep))
                ID_CLASS(id) = CL_ENTRY;
        }

        // if type of id st the deribed type, set tagname
        if (IS_STRUCT_TYPE(ID_TYPE(id))) {
            TYPE_TAGNAME(ID_TYPE(id)) = id;
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
 * Update the parent of types.
 */
static void
update_parent_type(HashTable * ht)
{
    HashEntry * e;
    HashSearch s;
    TYPE_ENTRY tep;
    TYPE_DESC tp;

    for (e = FirstHashEntry(ht, &s); e != NULL; e = NextHashEntry(&s)) {
        tep = GetHashValue(e);
        tp = tep->tp;
        if (TYPE_PARENT(tp) && tep->parent_type_id != NULL) {
            TYPE_ENTRY parent_tep = getTypeEntry(ht, tep->parent_type_id);
            if (parent_tep && TYPE_TAGNAME(parent_tep->tp) != NULL) {
                ID_SYM(TYPE_PARENT(tp)) = ID_SYM(TYPE_TAGNAME(parent_tep->tp));
            }
            free(tep->parent_type_id);
            tep->parent_type_id = NULL;
        }
    }
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

    update_parent_type(ht);

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
    char *is_module_specified = NULL;
    int ret = TRUE;

    if (!xmlMatchNode(reader,
                       XML_READER_TYPE_ELEMENT, "FmoduleProcedureDecl"))
        return FALSE;

    is_module_specified =
        (char *)xmlTextReaderGetAttribute(reader, BAD_CAST "is_module_specified");

    xmlSkipWhiteSpace(reader);

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
            EXT_ID new_ep;

            EXT_PROC_CLASS(tep->ep) = EP_MODULE_PROCEDURE;

            s = find_symbol(name);
            EXT_SYM(tep->ep) = s;

            if (EXT_PROC_TYPE(tep->ep) == NULL) {
                EXT_PROC_TYPE(tep->ep) = tep->tp;
            }
            FUNCTION_TYPE_SET_MOUDLE_PROCEDURE(tep->tp);
            new_ep = new_external_id(s);
            *new_ep = *tep->ep;
            EXT_NEXT(new_ep) = NULL;

            if (is_module_specified != NULL)
                EXT_PROC_IS_MODULE_SPECIFIED(new_ep) = TRUE;

            if (EXT_PROC_INTR_DEF_EXT_IDS(parent) == NULL)
                EXT_PROC_INTR_DEF_EXT_IDS(parent) = new_ep;
            else
                extid_put_last(EXT_PROC_INTR_DEF_EXT_IDS(parent), new_ep);
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
 * input <varDecl> node
 */
static int
input_varDecl(xmlTextReaderPtr reader, HashTable * ht, EXT_ID parent)
{
    SYMBOL s;
    TYPE_DESC tp;
    ID id;
    ID tail = NULL;
    expv v;

    if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "varDecl"))
        return FALSE;

    if (!input_name_with_type(reader, ht, FALSE, &s, &tp))
        return FALSE;

    id = XMALLOC(ID, sizeof(*id));
    ID_SYM(id) = s;
    ID_TYPE(id) = tp;

    ID_LINK_ADD(id, EXT_PROC_ID_LIST(parent), tail);

    if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "value")) {
        if (!input_value(reader, ht, &v))
            return FALSE;
        VAR_INIT_VALUE(id) = v;
    }

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
 * input <FuseDecl> node
 */
static int
input_FimportDecl(xmlTextReaderPtr reader)
{
    int depth;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FimportDecl"))
        return FALSE;

    if (!xmlTextReaderIsEmptyElement(reader)) {
        depth = xmlTextReaderDepth(reader);

        /* skip until corresponding close tag */
        if (!xmlSkipUntil(reader, XML_READER_TYPE_END_ELEMENT,"FimportDecl",
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
    char * is_defined_io = NULL;

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
        EXT_PROC_INTERFACE_CLASS(ep) = INTF_ASSIGNMENT;
        free(is_assignment);
    }

    is_defined_io = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_defined_io");
    if (is_defined_io != NULL) {
        if (strcmp("WRITE(FORMATTED)", is_defined_io) == 0) {
            EXT_PROC_INTERFACE_CLASS(ep) = INTF_GENERIC_WRITE_FORMATTED;
            name = "_write_formatted";
        } else if (strcmp("WRITE(UNFORMATTED)", is_defined_io) == 0) {
            EXT_PROC_INTERFACE_CLASS(ep) = INTF_GENERIC_WRITE_UNFORMATTED;
            name = "_write_unformatted";
        } else if (strcmp("READ(FORMATTED)", is_defined_io) == 0) {
            EXT_PROC_INTERFACE_CLASS(ep) = INTF_GENERIC_READ_FORMATTED;
            name = "_read_formatted";
        } else if (strcmp("READ(UNFORMATTED)", is_defined_io) == 0) {
            EXT_PROC_INTERFACE_CLASS(ep) = INTF_GENERIC_READ_UNFORMATTED;
            name = "_read_unformatted";
        } else {
            return FALSE;
        }
        EXT_IS_BLANK_NAME(ep) = FALSE;
        EXT_SYM(ep) = find_symbol(name);
        free(is_defined_io);
    }

    EXT_PROC_INTERFACE_INFO(ep) =
        XMALLOC(struct interface_info *, sizeof(struct interface_info));
    EXT_IS_BLANK_NAME(ep) = FALSE;
    EXT_PROC_CLASS(ep) = EP_INTERFACE;
    EXT_PROC_INTERFACE_CLASS(ep) = INTF_DECL;
    EXT_IS_OFMODULE(ep) = TRUE;

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
        } else if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT,
                                "FimportDecl")) {
            if (!input_FimportDecl(reader))
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

    ep = new_external_id(s);
    id = find_ident_head(s, id_list);
    if (id != NULL) {
        *ep = *(PROC_EXT_ID(id));
    } else {
        EXT_PROC_TYPE(ep) = tp;
    }
    EXT_IS_OFMODULE(ep) = TRUE;
    EXT_NEXT(ep) = NULL;

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
    char * is_defined_io = NULL;
    int interface_class = 0;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "FinterfaceDecl"))
        return FALSE;

    name = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "name");
    is_operator = (char *) xmlTextReaderGetAttribute(reader,
                               BAD_CAST "is_operator");
    is_assignment = (char *) xmlTextReaderGetAttribute(reader,
                                 BAD_CAST "is_assignment");
    is_defined_io = (char *) xmlTextReaderGetAttribute(reader, BAD_CAST "is_defined_io");


    if (is_defined_io == NULL && name != NULL) {
        id = find_ident_head(find_symbol(name), id_list);
        if (ID_CLASS(id) == CL_TAGNAME) { /* for multi class */
            id = find_ident_head(ID_SYM(id), ID_NEXT(id));
        }
        interface_class = INTF_OPERATOR;
        free(is_operator);
        free(name);
    } else if (is_defined_io != NULL) {
        if (strcmp("WRITE(FORMATTED)", is_defined_io) == 0) {
            interface_class = INTF_GENERIC_WRITE_FORMATTED;
            name = "_write_formatted";
        } else if (strcmp("WRITE(UNFORMATTED)", is_defined_io) == 0) {
            interface_class = INTF_GENERIC_WRITE_UNFORMATTED;
            name = "_write_unformatted";
        } else if (strcmp("READ(FORMATTED)", is_defined_io) == 0) {
            interface_class = INTF_GENERIC_READ_FORMATTED;
            name = "_read_formatted";
        } else if (strcmp("READ(UNFORMATTED)", is_defined_io) == 0) {
            interface_class = INTF_GENERIC_READ_UNFORMATTED;
            name = "_read_unformatted";
        } else {
            free(is_defined_io);
            return FALSE;
        }
        id = find_ident_head(find_symbol(name), id_list);
        free(is_defined_io);
    } else {
        assert(is_assignment != NULL); /* must be assignment */
        id = find_ident_head(find_symbol("="), id_list);
        interface_class = INTF_ASSIGNMENT;
        free(is_assignment);
    }

    ep = PROC_EXT_ID(id);
    EXT_PROC_INTERFACE_INFO(ep) =
        XMALLOC(struct interface_info *, sizeof(struct interface_info));
    EXT_PROC_INTERFACE_CLASS(ep) = interface_class;
    EXT_IS_BLANK_NAME(ep) = FALSE;
    EXT_PROC_CLASS(ep) = EP_INTERFACE;
    EXT_PROC_INTERFACE_CLASS(ep) = INTF_DECL;

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
 * input <declarations> node in .xmod
 */
static int
input_module_declarations(xmlTextReaderPtr reader, HashTable * ht,
			  struct module * mod)
{
    SYMBOL s;
    TYPE_DESC tp;
    ID id;
    expv v;

    if (!xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "declarations"))
        return FALSE;

    /* move to next node */
    xmlSkipWhiteSpace(reader);

    while (!xmlMatchNodeType(reader, XML_READER_TYPE_END_ELEMENT)) {
      if (!xmlExpectNode(reader, XML_READER_TYPE_ELEMENT, "varDecl"))
        return FALSE;

      if (!input_name_with_type(reader, ht, FALSE, &s, &tp))
        return FALSE;

      // search paramter symbol
      FOREACH_ID(id,mod->head){
	if(ID_SYM(id) == s) break;
      }
      if(id == NULL)  return FALSE;
      
      if (xmlMatchNode(reader, XML_READER_TYPE_ELEMENT, "value")) {
        if (!input_value(reader, ht, &v))
	  return FALSE;
        VAR_INIT_VALUE(id) = v;
      }
      if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "varDecl"))
        return FALSE;
    }

    if (!xmlExpectNode(reader, XML_READER_TYPE_END_ELEMENT, "declarations"))
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

static int
update_struct_type(HashTable * ht)
{
    HashEntry * e;
    HashSearch s;
    TYPE_ENTRY tep;
    TYPE_DESC tp;
    ID mem;

    for (e = FirstHashEntry(ht, &s); e != NULL; e = NextHashEntry(&s)) {
        tep = GetHashValue(e);
        tp = tep->tp;
        if (TYPE_BASIC_TYPE(tp) == TYPE_STRUCT) {
            FOREACH_TYPE_BOUND_GENERIC(mem, tp) {
                /*
                 * generic type bound procedure
                 */
                ID binding;
                ID bindto;
                FOREACH_ID(binding, TBP_BINDING(mem)) {
                    bindto = find_struct_member(tp, ID_SYM(binding));
                    if (bindto == NULL ||
                        ID_CLASS(bindto) != CL_TYPE_BOUND_PROC ||
                        TBP_BINDING_ATTRS(bindto) & TYPE_BOUND_PROCEDURE_IS_GENERIC) {
                        return FALSE;
                    }
                    ID_TYPE(binding) = ID_TYPE(bindto);
                }
                TYPE_BOUND_GENERIC_TYPE_GENERICS(ID_TYPE(mem)) = TBP_BINDING(mem);
            }
        }
    }

    return TRUE;
}


/**
 * input <OmniFortranModule> node
 */
static int
input_module(xmlTextReaderPtr reader, struct module * mod, int is_intrinsic)
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

    /* <declarations> node */
    input_module_declarations(reader,&ht,mod); /* optional */

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

    MODULE_IS_INTRINSIC(mod) = is_intrinsic;
    if (MODULE_IS_INTRINSIC(mod)) {
        /*
         * The parameters from the intrinsic module should not be expanded,
         * so remove initial value of thems.
         */
        ID id;
        FOREACH_ID(id, MODULE_ID_LIST(mod)) {
            VAR_INIT_VALUE(id) = NULL;
        }
    }

    /*
     * Update insuffcient types
     */
    if (!update_struct_type(&ht))
        return FALSE;

    free(version);
    return TRUE;
}

#include <stdlib.h>

#if defined(_FC_IS_GFORTRAN)
#define _XMPMOD_NAME "T_Module"
#define _XMPMOD_LEN 8
#elif defined(_FC_IS_FRTPX)
#define _XMPMOD_NAME "T_FJModule"
#define _XMPMOD_LEN 10
#endif

const char *
search_intrinsic_include_path(const char * filename)
{
    static char path[MAX_PATH_LEN];
    FILE * fp;

    if (xmoduleIncludeDirv) {
        strcpy(path, xmoduleIncludeDirv);
    } else {
        strcpy(path, DEFAULT_INSTRINSIC_XMODULES_PATH);
    }

    strcat(path, "/");
    strcat(path, filename);

    if ((fp = fopen(path, "r")) != NULL) {
        fclose(fp);
        return path;
    }

    return NULL;
}

/**
 * input data from the module intermediate file
 */
int
input_intermediate_file(const SYMBOL mod_name,
                        const SYMBOL submod_name,
                        struct module **pmod,
                        const char * extension)
{
    int ret;
    char filename[FILE_NAME_LEN];
    const char * filepath;
    xmlTextReaderPtr reader;
    int is_intrinsic = FALSE;

    if (mod_name == NULL || pmod == NULL) {
        return FALSE;
    }

    /* search for "xxx.xmod" */
    bzero(filename, sizeof(filename));
    if (!submod_name) {
        snprintf(filename, sizeof(filename), "%s.%s",
                 SYM_NAME(mod_name),
                 extension);
    } else {
        snprintf(filename, sizeof(filename), "%s:%s.%s",
                 SYM_NAME(mod_name), SYM_NAME(submod_name),
                 extension);
    }

    filepath = search_include_path(filename);
    reader = xmlNewTextReaderFilename(filepath);

#if defined(_FC_IS_GFORTRAN) || defined(_FC_IS_FRTPX)
    // if not found, then search for "xxx.mod" and convert it into "xxx.xmod"
    if (reader == NULL){

        char command2[6 + _XMPMOD_LEN];
        bzero(command2, 6 + _XMPMOD_LEN);
        strcpy(command2, "which ");
	strcat(command2, _XMPMOD_NAME);
        if (system(command2) != 0){
	  warning("No module translator found.");
	  return FALSE;
	}

        char filename2[FILE_NAME_LEN];
        const char * filepath2;

        bzero(filename2, sizeof(filename));
        strcpy(filename2, SYM_NAME(mod_name));
        strcat(filename2, ".mod");
        filepath2 = search_include_path(filename2);

        if (!filepath2) return FALSE;

        char command[FILE_NAME_LEN + 9];
        bzero(command, sizeof(filename2) + 9);
        strcpy(command, _XMPMOD_NAME);
        strcat(command, " ");
        strcat(command, filepath2);
        if (system(command) != 0) return FALSE;

	filepath = search_include_path(filename);
        reader = xmlNewTextReaderFilename(filepath);
    }
#endif

    if (reader == NULL) {
        filepath = search_intrinsic_include_path(filename);
        reader = xmlNewTextReaderFilename(filepath);
        is_intrinsic = TRUE;
    }

    if (reader == NULL)
        return FALSE;

    *pmod = XMALLOC(struct module *, sizeof(struct module));
    MODULE_NAME(*pmod) = mod_name;

    ret = input_module(reader, *pmod, is_intrinsic);

    xmlTextReaderClose(reader);

    // (*pmod)->filepath = strdup(filepath);

    return ret;
}
