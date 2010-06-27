/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file xcodeml-traverse.c
 */

#include "xcodeml.h"

static int enter_XcodeProgram(XcodeMLNode * xcodeProgram);
static int enter_typeTable(XcodeMLNode * node);
static int enter_module(XcodeMLNode * node);
static int enter_symbols(XcodeMLNode * node);
static int enter_declarations(XcodeMLNode * node);
static int enter_declarations_wsymbol(XcodeMLNode * node, char * symbol);
static int enter_useDecl(XcodeMLNode * node);
static int enter_useOnlyDecl(XcodeMLNode * node);
static int enter_varDecl(XcodeMLNode * node);
static int enter_structDecl(XcodeMLNode * node);
static int enter_taggedType(char * type_sig, char * tagname);

static int enter_interfaceDecl(XcodeMLNode * node);
static int enter_functionDecl(XcodeMLNode * node);
static int enter_params(XcodeMLNode * node);

static int clear_contains(XcodeMLNode * node);
static int enter_contains(XcodeMLNode * node);
static int enter_functionDefinition(XcodeMLNode * node);

int is_inner_module;

struct module_list {
    char * filename; /* xmod file name*/
    XcodeMLNode * xcodeml;
    struct module_list * next;
};

struct module_list * module_list_head = NULL;
struct module_list * module_list_tail = NULL;


#define MODULE_FILENAME(module_list) ((module_list)->filename)
#define MODULE_XCODEML(module_list) ((module_list)->xcodeml)
#define MODULE_LIST_ISEMPTY (module_list_head == NULL)
#define MODULE_LINK(module_list) ((module_list)->next)
#define MODULE_LINK_ADD(module_list) \
    { if(module_list_head == NULL) {     \
        module_list_head = module_list;  \
        module_list_tail = module_list;  \
    } else {                                         \
        MODULE_LINK(module_list_tail) = module_list; \
        module_list_tail = module_list;              \
    }}


static void
add_symbol(XcodeMLNode *nameNode)
{
    if(nameNode == NULL)
        return;

    XcodeMLList *lp, *l = (XcodeMLList*)malloc(sizeof(XcodeMLList));
    XcodeMLNode *valNode = NULL;
    FOR_ITEMS_IN_XCODEML_LIST(lp, nameNode) {
        if(XCODEML_TYPE(XCODEML_LIST_NODE(lp)) == XcodeML_Value) {
            valNode = XCODEML_LIST_NODE(lp);
            break;
        }
    }
    if(valNode == NULL)
        return;
    XCODEML_LIST_NODE(l) = valNode;

    if(current_symbol_stack->id_list != NULL)
        XCODEML_LIST_NEXT(l) = current_symbol_stack->id_list;
    else
        XCODEML_LIST_NEXT(l) = NULL;

    current_symbol_stack->id_list = l;
}


void
push_symbol_stack(XcodeMLNode *decls)
{
    symbol_stack *st = (symbol_stack*)malloc(sizeof(symbol_stack));
    memset(st, 0, sizeof(*st));
    if (current_symbol_stack != NULL)
        st->next = current_symbol_stack->next;
    current_symbol_stack = st;

    XcodeMLList *lp, *lp1;

    FOR_ITEMS_IN_XCODEML_LIST(lp, decls) {
        XcodeMLNode *n = XCODEML_LIST_NODE(lp);
        if (n == NULL)
            continue;
        const char *tag = XCODEML_NAME(n);
        if(tag == NULL)
            continue;
        if (strcmp(tag, "varDecl") == 0) {
            add_symbol(GET_NAME(n));
        } else if (strcmp(tag, "FinterfaceDecl") == 0) {
            FOR_ITEMS_IN_XCODEML_LIST(lp1, n) {
                XcodeMLNode *n1 = XCODEML_LIST_NODE(lp1);
                if (strcmp(XCODEML_NAME(n1), "FfunctionDecl") == 0) {
                    add_symbol(GET_NAME(n1));
                }
            }
        }
    }
}


void
pop_symbol_stack()
{
    if(current_symbol_stack == NULL)
        return;

    symbol_stack *next = current_symbol_stack->next;
    XcodeMLList *lp, *lpn;

    for(lp = current_symbol_stack->id_list,
        lpn = (lp ? XCODEML_LIST_NEXT(lp) : NULL); lp != NULL;
        lp = lpn, lpn = (lp ? XCODEML_LIST_NEXT(lp) : NULL)) {
        free(lp);
    }

    free(current_symbol_stack);

    current_symbol_stack = next;
}


int
xcodeml_has_symbol(const char *symbol)
{
    XcodeMLList *lp;
    if(current_symbol_stack == NULL)
        return FALSE;
    
    for(lp = current_symbol_stack->id_list; lp != NULL; lp = XCODEML_LIST_NEXT(lp)) {
        XcodeMLNode *n = XCODEML_LIST_NODE(lp);
        const char *name = XCODEML_VALUE(n);
        if(strcmp(symbol, name) == 0)
            return TRUE;
    }

    return FALSE;
}



/**
 * Reads a module definition from XcodeML and writes the declarations inside the module.
 *
 * @param module_filename the file name of XcodeML.
 * @param fortran_filename the file name wrriten the declarations inside the module.
 * @return returns 0 if fail to read XcodeML, otherwise returns 1.
 */
int
use_module(const char * module_filename, const char * fortran_filename)
{
    XcodeMLNode * xcodeProgram = NULL;
    struct module_list * lp;
    FILE * outFd;

    outFd = fopen(fortran_filename, "w");
    if (outFd == NULL) {
        fprintf(stderr, "cannot open file %s\n", fortran_filename);
        return FALSE;
    }

    for(lp = module_list_head; lp != NULL; lp = MODULE_LINK(lp)) {
        if(lp == NULL)
            break;
        if(strcmp(MODULE_FILENAME(lp), module_filename) == 0) {
            xcodeProgram = MODULE_XCODEML(lp);
            break;
        }
    }

    if(xcodeProgram == NULL) {
        char * filename;
        xcodeProgram = xcodeml_ParseFile(module_filename);
        if (xcodeProgram == NULL) {
            return FALSE;
        }
        lp = XMALLOC(struct module_list * , sizeof(struct module_list));
        filename = XMALLOC(char *, strlen(module_filename) + 1);
        strcpy(filename, module_filename);
        MODULE_FILENAME(lp) = filename;
        MODULE_XCODEML(lp) = xcodeProgram;
        MODULE_LINK_ADD(lp);
    }

    init_outputf(outFd);
    enter_XcodeProgram(xcodeProgram);
    fclose(outFd);
    return TRUE;
}

/**
 * Reads XcodeML as module definition and writes the declarations inside the module.
 *
 * @param module_filename the file name of XcodeML.
 * @param outFd the file discripter wrriten the declarations inside the module.
 * @return returns 1 if fail to read XcodeML, otherwise returns 0.
 */
int
use_module_to(const char * module_filename, FILE * outFd)
{
    XcodeMLNode * xcodeProgram;
    symbol_filter * filter;

    xcodeProgram = xcodeml_ParseFile(module_filename);
    if (xcodeProgram == NULL) {
        return 0;
    }

    filter = push_new_filter();
    FILTER_USAGE(filter) = RENAME;

    init_outputf(outFd);

    enter_XcodeProgram(xcodeProgram);

    return 1;
}


int
enter_XcodeProgram(XcodeMLNode * xcodeProgram)
{
    XcodeMLNode * typeTable;
    XcodeMLNode * globalDeclaration;
    XcodeMLNode * module;

    if(xcodeProgram == NULL) /* parse error. */
        return FALSE;

    typeTable = GET_TYPETABLE(xcodeProgram);

    enter_typeTable(typeTable);

    globalDeclaration = GET_GLOBALDECLS(xcodeProgram);
    module = GET_FMODULEDEFINITION(globalDeclaration);

    enter_module(module);

    return TRUE;
}


int
enter_typeTable(XcodeMLNode * typeTable)
{
    XcodeMLList * lp;

    if (typeTable == NULL ||
        (XCODEML_TYPE(typeTable) != XcodeML_Element))
        return FALSE;

    typetable_init();
    FOR_ITEMS_IN_XCODEML_LIST(lp, typeTable) {
        XcodeMLNode * t = XCODEML_LIST_NODE(lp);
        if (t == typeTable)
            continue;

        if (XCODEML_TYPE(t) == XcodeML_Element)
            typetable_enhash(t);
    }

    return TRUE;
}


int
enter_module(XcodeMLNode * module)
{
    XcodeMLNode * symbols, * declarations;
    is_inner_module = TRUE;

    if (module == NULL)
        return FALSE;

    symbols = GET_SYMBOLS(module);
    declarations = GET_DECLARATIONS(module);
    containsStmt = GET_CONTAINS(module);

    clear_contains(containsStmt);
    enter_symbols(symbols);
    push_symbol_stack(declarations);
    enter_declarations(declarations);
    pop_symbol_stack();
    enter_contains(containsStmt);

    return TRUE;
}


int
enter_symbols(XcodeMLNode * symbols)
{
    XcodeMLList * lp;

    if (symbols == NULL)
        return FALSE;

    FOR_ITEMS_IN_XCODEML_LIST(lp, symbols) {
        XcodeMLNode * x;
        char * name, * type, * sclass;

        x = XCODEML_LIST_NODE(lp);

        if (XCODEML_NAME(x) == NULL ||
            XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if (strcmp(XCODEML_NAME(x), "id") != 0) {
            continue;
        }

        name = GET_ID_NAME(x);
        type = GET_ID_TYPE(x);
        sclass = GET_ID_SCLASS(x);

        if (name == NULL ||
            type == NULL ||
            sclass == NULL)
            continue;

        if (strcmp(sclass, "ftype_name") == 0) {
            xentry * xe;

            xe = typetable_dehash(type);
            if (xe == NULL || GET_CONTENT(xe) == NULL)
                continue;

            GET_TAGNAME(xe) = name;
        }
    }
    return TRUE;
}

int
enter_declarations(XcodeMLNode * declarations)
{
    enter_declarations_wsymbol(declarations, NULL);
    return TRUE;
}

int
enter_declarations_wsymbol(XcodeMLNode * declarations, char * symbol)
{
    XcodeMLList * lp;

    if (declarations == NULL)
        return FALSE;

    FOR_ITEMS_IN_XCODEML_LIST(lp, declarations) {
        XcodeMLNode * x;
        x = XCODEML_LIST_NODE(lp);

        if (XCODEML_NAME(x) == NULL ||
            XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if (strcmp(XCODEML_NAME(x), "FuseDecl") == 0) {
            enter_useDecl(x);
            continue;
        }

        if (strcmp(XCODEML_NAME(x), "FuseOnlyDecl") == 0) {
            enter_useOnlyDecl(x);
            continue;
        }

        if (strcmp(XCODEML_NAME(x), "varDecl") == 0) {
            if(enter_varDecl(x) == FALSE && symbol != NULL) {
                if(strcmp(symbol, xcodeml_getAsString(GET_NAME(x)))== 0) {
                    if(outf_decl(GET_TYPE(GET_NAME(x)), symbol, NULL, false, true))
                        outf_flush();
                }
            }
            continue;
        }

        if (strcmp(XCODEML_NAME(x), "FstructDecl") == 0) {
            enter_structDecl(x);
            continue;
        }

        if (strcmp(XCODEML_NAME(x), "FinterfaceDecl") == 0) {
            enter_interfaceDecl(x);
            continue;
        }
    }

    return TRUE;
}

int
enter_contains(XcodeMLNode * containsStmt)
{
    XcodeMLList * lp;

    if (containsStmt == NULL)
        return FALSE;

    FOR_ITEMS_IN_XCODEML_LIST(lp, containsStmt) {
        XcodeMLNode * x;
        x = XCODEML_LIST_NODE(lp);

        if (XCODEML_NAME(x) == NULL ||
            XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if (strcmp(XCODEML_NAME(x), "FfunctionDefinition") == 0) {
            enter_functionDefinition(x);
            continue;
        }
    }
    return TRUE;
}


static int
clear_contains(XcodeMLNode * containsStmt)
{
    XcodeMLList * lp;

    if (containsStmt == NULL)
        return FALSE;

    FOR_ITEMS_IN_XCODEML_LIST(lp, containsStmt) {
        XcodeMLNode * x;
        x = XCODEML_LIST_NODE(lp);

        if (XCODEML_NAME(x) == NULL ||
            XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if (strcmp(XCODEML_NAME(x), "FfunctionDefinition") == 0) {
            XCODEML_IS_OUTPUTED(x) = FALSE;
            continue;
        }
    }
    return TRUE;
}


void
outf_filter_use(symbol_filter *filter)
{
    symbol_filterElem * filterElem;
    const char *use;
    int i = 0;

    for(filterElem = filter->list; filterElem != NULL; filterElem = filterElem->next) {
        if (filterElem->use == NULL)
            break;
        use = filterElem->use;

        if(xcodeml_has_symbol(use) == FALSE) {
            /* symbol is not in current context,
               so symbol is expected to be in the module. */
            if(i != 0) {
                outf_token(",");
            }
            i++;

            if (filterElem->local != NULL) {
                outf_token(filterElem->local);
                outf_token("=>");
            }
            outf_token(use);
        }
    }
}


int
enter_useDecl(XcodeMLNode * useDecl)
{
    XcodeMLNode * nameAttr;
    XcodeMLList * lp;
    char * moduleName;
    symbol_filter * filter = peek_filter();
    int i = 0;

    nameAttr = GET_NAME(useDecl);

    moduleName = xcodeml_getAsString(nameAttr);

    outf_token("USE ");
    outf_token(moduleName);

    if(FILTER_USAGE(filter) == LIMIT) {
        outf_token(", ONLY: ");
        outf_filter_use(filter);
    } else {
        FOR_ITEMS_IN_XCODEML_LIST(lp, useDecl) {
            XcodeMLNode * x = XCODEML_LIST_NODE(lp);
            char * local, * use;

            if (x == useDecl)
                continue;

            if (XCODEML_TYPE(x) != XcodeML_Element)
                continue;

            local = apply_filter(filter, GET_LOCAL(x));
            use = apply_filter(filter, GET_USE(x));

            if (local == NULL && use == NULL)
                continue;

            if(i != 0) {
                outf_token(",");
            }
            i++;

            if (local != NULL) {
                outf_token(local);
                outf_token("=>");
            }
            outf_token(use);
        }
    }
    outf_flush();
    return TRUE;
}


int
enter_useOnlyDecl(XcodeMLNode * useOnlyDecl)
{
    XcodeMLNode * nameAttr;
    XcodeMLList * lp;
    char * moduleName;
    symbol_filter * filter = peek_filter();
    int i = 0;

    nameAttr = GET_NAME(useOnlyDecl);

    moduleName = xcodeml_getAsString(nameAttr);

    outf_token("USE ");
    outf_token(moduleName);
    outf_token(", ONLY: ");

    FOR_ITEMS_IN_XCODEML_LIST(lp, useOnlyDecl) {
        XcodeMLNode * x = XCODEML_LIST_NODE(lp);
        char * local, * use;

        if (x == useOnlyDecl)
            continue;

        if (XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        local = apply_filter(filter, GET_LOCAL(x));
        use = apply_filter(filter, GET_USE(x));
        
        if (local == NULL && use == NULL)
            continue;

        if(i != 0) {
            outf_token(",");
        }
        i++;

        if (local != NULL) {
            outf_token(local);
            outf_token("=>");
        }
        outf_token(use);
    }
    outf_flush();
    return TRUE;
}


int
enter_varDecl(XcodeMLNode * varDecl)
{
    XcodeMLNode * name;
    XcodeMLNode * value;
    char * type_signature, * symbol, * orgSymbol;
    symbol_filter * filter = peek_filter();
    int ret;

    name = GET_NAME(varDecl);
    value = GET_VALUE(varDecl);
    type_signature = GET_TYPE(name);

    orgSymbol = xcodeml_getAsString(name);

    if (type_signature == NULL) {
        /* error! need type attribute. */
        return FALSE;
    }

    symbol = apply_filter(filter, orgSymbol);

    if (symbol == NULL) {
        symbol = orgSymbol;
    }

    ret = outf_decl(type_signature, symbol, value, true, false);

    outf_flush();
    return ret;
}


int
enter_structDecl(XcodeMLNode * structDecl)
{
    XcodeMLNode * name;
    char * symbol, * type_sig;

    if (structDecl == NULL)
        return FALSE;

    name = GET_NAME(structDecl);
    type_sig = GET_TYPE(name);
    symbol = xcodeml_getAsString(name);

    outf_token("TYPE ");
    outf_tokenln(symbol);
    enter_taggedType(type_sig, symbol);
    outf_tokenln("END TYPE");
    return TRUE;
}


int
enter_taggedType(char * type_sig, char * tagname)
{
    xentry * xe = NULL;
    XcodeMLNode * structType, * symbol;
    XcodeMLList * lp;

    xe = typetable_dehash(type_sig);

    if (xe == NULL)
        return FALSE;

    GET_TAGNAME(xe) = tagname;
    structType = GET_CONTENT(xe);
    symbol = GET_SYMBOLS(structType);

    if(XCODEML_TYPE(symbol) != XcodeML_Element)
        return FALSE;

    if(GET_IS_SEQUENCE(structType))
        outf_tokenln("SEQUENCE");

    FOR_ITEMS_IN_XCODEML_LIST(lp, symbol) {
        XcodeMLNode * x = XCODEML_LIST_NODE(lp);

        if(XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if (strcmp(XCODEML_NAME(x), "id") == 0) {
            XcodeMLNode * name = GET_NAME(x);
            char * type = GET_TYPE(x);
            char * symbol = xcodeml_getAsString(name);

            outf_decl(type, symbol, NULL, false, true);
            outf_flush();
            continue;
        }
    }
    return TRUE;
}

int
enter_interfaceDecl(XcodeMLNode * intfDecl)
{
    char * name;
    char interface_spec[MAXBUFFER];
    int name_flag = 0;
    XcodeMLList * lp;

    if (intfDecl == NULL)
        return FALSE;
    is_inner_module = FALSE;

    name = GET_INTF_NAME(intfDecl);

    outf_token("INTERFACE ");
    if (name != NULL) {
        if (GET_INTF_IS_OPERATOR(intfDecl)) {
            sprintf(interface_spec, "OPERATOR(%s)", name);
        } else if (GET_INTF_IS_ASSGIN(intfDecl)) {
#if 0
            sprintf(interface_spec, "ASSIGNMENT(%s)", name);
#else
            sprintf(interface_spec, "ASSIGNMENT(=)");
#endif
        } else {
            sprintf(interface_spec, "%s", name);
            name_flag = 1;
        }
        outf_token(interface_spec);
    }
    outf_flush();

    FOR_ITEMS_IN_XCODEML_LIST(lp, intfDecl) {
        XcodeMLNode * x = XCODEML_LIST_NODE(lp);
        if (x == intfDecl)
            continue;

        if (strcmp(XCODEML_NAME(x), "FfunctionDecl") == 0) {
            enter_functionDecl(x);

        } else if (strcmp(XCODEML_NAME(x), "FmoduleProcedureDecl") == 0) {
            XcodeMLList * llp;

            FOR_ITEMS_IN_XCODEML_LIST(llp, x) {
                XcodeMLNode * funcDef, * xx = XCODEML_LIST_NODE(llp);
                char * funcName;

                if (xx == x)
                    continue;

                if (strcmp(XCODEML_NAME(xx), "name") != 0)
                    continue;

                funcName = xcodeml_getAsString(xx);
                if(funcName == NULL)
                    continue;

                funcDef = get_funcDef(containsStmt, funcName);
                if(funcDef != NULL) {
                    enter_functionDecl(funcDef);
                } else {
                    outf_token("MODULE PROCEDURE ");
                    outf_tokenln(funcName);
                }
            }
        }
    }

    outf_token("END INTERFACE ");
    if(name != NULL && name_flag == 1)
        outf_token(interface_spec);
    outf_flush();
    is_inner_module = TRUE;
    return TRUE;
}

int
enter_functionDecl(XcodeMLNode * funcDecl)
{
    char * funcName;
    char * type_sig;
    char * ref = NULL;
    char * result = NULL;
    xentry * entry;
    XcodeMLNode * symbols;
    XcodeMLNode * funcType;
    XcodeMLNode * params;
    XcodeMLNode * declarations;

    /* only for function definition. */
    symbols = GET_SYMBOLS(funcDecl);
    enter_symbols(symbols);

    type_sig = GET_TYPE(GET_NAME(funcDecl));
    funcName = xcodeml_getAsString(GET_NAME(funcDecl));

    entry = typetable_dehash(type_sig);
    if (IS_FFUNCTIONTYPE(entry) == FALSE) {
        return FALSE;
    }

    funcType = GET_CONTENT(entry);

    if (GET_IS_PRIVATE(funcType) ||
        apply_filter(peek_filter(), funcName) == NULL)
        return TRUE;

#if 0
    if (GET_IS_PRIVATE(funcType)) {
        funcName = convert_to_non_use_symbol(funcName);
    }
#endif
    params = GET_PARAMS(funcType);
    ref = GET_RETURN(GET_CONTENT(entry));
    result = GET_RESULT_NAME(funcType);
    declarations = GET_DECLARATIONS(funcDecl);

    if(strcmp("Fvoid", ref) == 0) {
        outf_token("SUBROUTINE ");
    } else {
        outf_token("FUNCTION ");
    }
    outf_token(funcName);

    outf_token("(");
    enter_params(params);
    outf_token(")");

    if(result != NULL) {
        outf_token(" result(");
        outf_token(result);
        outf_token(")");
    }

    outf_flush();

    enter_declarations_wsymbol(declarations,result?:funcName);

    if(strcmp("Fvoid", ref) == 0) {
        outf_token("END SUBROUTINE");
    } else {
        outf_token("END FUNCTION");
    }
    outf_flush();

    XCODEML_IS_OUTPUTED(funcDecl) = TRUE;
    return TRUE;
}

int
enter_functionDefinition(XcodeMLNode * funcDef)
{
#if 0
    is_inner_module = FALSE;
    if(XCODEML_IS_OUTPUTED(funcDef) == FALSE) {
        char * type_sig = GET_TYPE(GET_NAME(funcDef));
        XcodeMLNode * funcType = typetable_dehash(type_sig);
        if (funcType == NULL || GET_IS_PRIVATE(funcType))
            return FALSE;
        if (IS_FFUNCTIONTYPE(funcType)) {
            outf_tokenln("INTERFACE");
            enter_functionDecl(funcDef);
            outf_tokenln("END INTERFACE");
        }
    }
    is_inner_module = TRUE;
#else
    char * type_sig = GET_TYPE(GET_NAME(funcDef));
    xentry * xe = typetable_dehash(type_sig);

    XcodeMLNode * funcType = NULL;
    char * funcName = NULL;

    if (xe == NULL || !IS_FFUNCTIONTYPE(xe))
        return FALSE;

    funcType = GET_CONTENT(xe);
    if (funcType == NULL || GET_IS_PRIVATE(funcType))
        return FALSE;

    funcName = xcodeml_getAsString(GET_NAME(funcDef));
    funcName = apply_filter(peek_filter(),funcName);
    if (funcName == NULL)
        return FALSE;
    outf_tokenln("INTERFACE");
    outf_token("MODULE PROCEDURE ");
    outf_tokenln(funcName);
    outf_tokenln("END INTERFACE");
#endif
    return TRUE;
}

int
enter_params(XcodeMLNode * params)
{
    XcodeMLList * lp;

    FOR_ITEMS_IN_XCODEML_LIST(lp, params) {
        XcodeMLNode * x = XCODEML_LIST_NODE(lp);
        char * name;

        if (x == params)
            continue;

        if (XCODEML_NAME(x) == NULL)
            break;

        if (strcmp(XCODEML_NAME(x), "name") != 0)
            continue;
        name = xcodeml_getAsString(x);

        if(is_use_symbol(name) == false)
            name = convert_to_non_use_symbol(name);
        else {
            name = apply_filter(peek_filter(),name)?:name;
        }

        outf_token(name);

        if(XCODEML_LIST_NEXT(lp) != NULL)
            outf_token(",");
    }
    return TRUE;
}
