/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "config.h"
#include "exc_platform.h"

#include <libxml/parser.h>
#include <libxml/tree.h>
#include <assert.h>

#ifndef __cplusplus
typedef enum {
    false = 0,
    true = 1
} bool;
#endif /* ! __cplusplus */

#include "xcodeml-node.h"

#define MAXBUFFER 131
#define LAST_COLUMN MAXBUFFER - 1

extern char *   xmalloc _ANSI_ARGS_((int size));
#define XMALLOC(type, size) ((type)xmalloc(size))

extern int module_debug_flag;
extern int is_inner_module;

extern char *           xcodeml_GetAttributeValue(XcodeMLNode *ndPtr);
extern char *           xcodeml_GetElementValue(XcodeMLNode *ndPtr);

extern XcodeMLNode *    xcodeml_ParseFile(const char *file);
extern void             xcodeml_DumpTree(FILE *fd, XcodeMLNode *ndPtr,
                                         int lvl);

extern XcodeMLNode * xcodeml_getByName(XcodeMLNode *ndPtr, char * name);
extern XcodeMLNode * xcodeml_getAttrByName(XcodeMLNode *ndPtr, char * name);

extern XcodeMLNode * xcodeml_getElem(XcodeMLNode *ndPtr, int order);

#define GET_CHILD0(ndPtr) xcodeml_getElem(ndPtr, 0)
#define GET_CHILD1(ndPtr) xcodeml_getElem(ndPtr, 1)
#define GET_CHILD2(ndPtr) xcodeml_getElem(ndPtr, 2)

extern bool xcodeml_getAsBool(XcodeMLNode * ndPtr);
extern char * xcodeml_getAsString(XcodeMLNode * ndPtr);

/* macros for a xcodeProgram tag */
#define GET_TYPETABLE(ndPtr)         xcodeml_getByName((ndPtr), "typeTable")
#define GET_GLOBALSYMBOLS(ndPtr)     xcodeml_getByName((ndPtr), "globalSymbols")
#define GET_GLOBALDECLS(ndPtr)       xcodeml_getByName((ndPtr), "globalDeclarations")

/* macros for a id tag. */
#define GET_ID_TYPE(ndPtr)           xcodeml_getAsString(xcodeml_getByName((ndPtr), "type"))
#define GET_ID_SCLASS(ndPtr)         xcodeml_getAsString(xcodeml_getByName((ndPtr), "sclass"))
#define GET_ID_NAME(ndPtr)           xcodeml_getAsString(xcodeml_getByName((ndPtr), "name"))

/* macros for useDecl/useOnlyDecl */
#define GET_LOCAL(ndPtr)             xcodeml_getAsString(xcodeml_getByName((ndPtr), "local_name"))
#define GET_USE(ndPtr)               xcodeml_getAsString(xcodeml_getByName((ndPtr), "use_name"))

/* macros for a type tag. */
#define GET_TYPE(ndPtr)              xcodeml_getAsString(xcodeml_getByName((ndPtr), "type"))
#define GET_REF(ndPtr)               xcodeml_getAsString(xcodeml_getByName((ndPtr), "ref"))

#define GET_LEN(ndPtr)               xcodeml_getByName((ndPtr), "len")
#define GET_KIND(ndPtr)              xcodeml_getByName((ndPtr), "kind")

#define GET_IS_PUBLIC(ndPtr)         xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_public"))
#define GET_IS_PRIVATE(ndPtr)        xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_private"))
#define GET_IS_POINTER(ndPtr)        xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_pointer"))
#define GET_IS_TARGET(ndPtr)         xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_target"))
#define GET_IS_EXTERNAL(ndPtr)       xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_external"))
#define GET_IS_INTRINSIC(ndPtr)      xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_intrinsic"))
#define GET_IS_OPTIONAL(ndPtr)       xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_optional"))
#define GET_IS_SAVE(ndPtr)           xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_save"))
#define GET_IS_PARAMETER(ndPtr)      xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_parameter"))
#define GET_IS_ALLOCATABLE(ndPtr)    xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_allocatable"))
#define GET_IS_SEQUENCE(ndPtr)       xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_sequence"))
#define GET_INTENT(ndPtr)            xcodeml_getAsString(xcodeml_getByName((ndPtr), "intent"))

/* macros for module tag */
#define GET_FMODULEDEFINITION(ndPtr) xcodeml_getByName((ndPtr), "FmoduleDefinition")

#define GET_SYMBOLS(ndPtr)           xcodeml_getByName((ndPtr), "symbols")
#define GET_DECLARATIONS(ndPtr)      xcodeml_getByName((ndPtr), "declarations")
#define GET_CONTAINS(ndPtr)          xcodeml_getByName((ndPtr), "FcontainsStatement")

/* macros for declaraions */
#define GET_NAME(ndPtr)              xcodeml_getByName((ndPtr), "name")
#define IS_TAGNAME_OF(entry, name) \
    ((entry) && GET_CONTENT(entry) && XCODEML_NAME(GET_CONTENT(entry)) && \
    (strcmp((name), XCODEML_NAME(GET_CONTENT(entry))) == 0))
#define IS_FFUNCTIONTYPE(entry)      IS_TAGNAME_OF((entry), "FfunctionType")
#define IS_FSTRUCTTYPE(entry)      IS_TAGNAME_OF((entry), "FstructType")


/* macros for varDecl */
#define GET_VALUE(ndPtr)             xcodeml_getByName((ndPtr), "value")

/* macros for expression */
#define GET_ARGUMENTS(ndPtr)         xcodeml_getByName((ndPtr), "arguments")
#define GET_MEMBER(ndPtr)            xcodeml_getAsString(xcodeml_getByName((ndPtr), "member"))

#define GET_UPPER(ndPtr)             xcodeml_getByName((ndPtr), "upperBound")
#define GET_LOWER(ndPtr)             xcodeml_getByName((ndPtr), "lowerBound")
#define GET_STEP(ndPtr)              xcodeml_getByName((ndPtr), "step")

#define GET_VAR(ndPtr)               xcodeml_getByName((ndPtr), "Var")
#define GET_VARREF(ndPtr)            xcodeml_getByName((ndPtr), "varRef")
#define GET_INDEXRANGE(ndPtr)        xcodeml_getByName((ndPtr), "indexRange")

#define GET_IS_ASHAPE(ndPtr)         xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_assumed_shape"))
#define GET_IS_ASIZE(ndPtr)          xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_assumed_size"))

/* macros for interface */
#define GET_INTF_NAME(ndPtr)         xcodeml_getAsString(xcodeml_getByName((ndPtr), "name"))
#define GET_INTF_IS_OPERATOR(ndPtr)  xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_operator"))
#define GET_INTF_IS_ASSGIN(ndPtr)    xcodeml_getAsBool(xcodeml_getByName((ndPtr), "is_assignment"))

/* macros for function */
#define GET_PARAMS(ndPtr)            xcodeml_getByName((ndPtr), "params")
#define GET_RETURN(ndPtr)            xcodeml_getAsString(xcodeml_getByName((ndPtr), "return_type"))
#define GET_RESULT_NAME(ndPtr)       xcodeml_getAsString(xcodeml_getByName((ndPtr), "result_name"))

struct xcodeml_type_entry {
    XcodeMLNode * content;
    struct xcodeml_type_entry * next;
    char * tagname;
};
#define GET_CONTENT(x) ((x)->content)
#define GET_NEXT(x)    ((x)->next)
#define GET_TAGNAME(x) ((x)->tagname)

typedef struct xcodeml_type_entry xentry;

extern void typetable_enhash(XcodeMLNode * type);
extern xentry * typetable_dehash(char * type_signature);

extern void typetable_init();
extern bool type_isPrimitive(char * type_signature);

/* for output */
extern void init_outputf(FILE * fd);
extern void outf_token(const char * token);
extern void outf_tokenln(const char * token);
extern void outf_flush();
extern int  outf_decl(char * type_signature, char * symbol,
                      XcodeMLNode * node, bool convertSymbol, int force);

extern XcodeMLNode * containsStmt;
extern XcodeMLNode * get_funcDef(XcodeMLNode * defs, char * name);
extern int xcodeml_has_symbol(const char *symbol);


struct priv_parm_list {
    char * symbol;
    struct priv_parm_list * next;
};

extern struct priv_parm_list * priv_parm_list_head;
extern struct priv_parm_list * priv_parm_list_tail;

typedef struct symbol_stack {
    XcodeMLList *id_list;
    struct symbol_stack * next;
} symbol_stack;

symbol_stack *current_symbol_stack;


#define PRIV_PARM_SYM(priv_parm_list) ((priv_parm_list)->symbol)
#define PRIV_PARM_LINK(priv_parm_list) ((priv_parm_list)->next)
#define PRIV_PARM_LINK_ADD(priv_parm_list) \
    { if(priv_parm_list_head == NULL) {     \
        priv_parm_list_head = priv_parm_list;  \
        priv_parm_list_tail = priv_parm_list;  \
    } else {                                         \
        PRIV_PARM_LINK(priv_parm_list_tail) = priv_parm_list; \
        priv_parm_list_tail = priv_parm_list;              \
    }}
#define PRIV_PARM_LINK_FOR(lp, priv_parm_list) \
    if (priv_parm_list_head != NULL) \
    for(lp = priv_parm_list_head;   \
        lp != NULL;\
        lp = PRIV_PARM_LINK(lp))

#include "xcodeml-module.h"
