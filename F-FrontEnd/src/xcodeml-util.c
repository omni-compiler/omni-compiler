/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "xcodeml.h"

static int          nonUseSymbolPrefixSequence = 0;
XcodeMLNode * containsStmt = NULL;

/**
 * Gets a content of XcodeMLNode as bool.
 */
bool
xcodeml_getAsBool(XcodeMLNode * ndPtr)
{
    char * content;

    if (ndPtr == NULL) {
        return false;
    }

    if (XCODEML_TYPE(ndPtr) != XcodeML_Value) {
        /* ERROR */
        /* may invalid usage. */
        return false;
    }

    content = XCODEML_VALUE(ndPtr);

    if (strlen(content) > 0) {
        if (strcmp("1",content) == 0 || strcmp("true",content) == 0) {
                return true;
        }

        if (strcmp("0",content) == 0 || strcmp("false",content) == 0) {
                return false;
        }
    }

    return false;
}

/**
 * Gets a content of XcodeMLNode as string.
 */
char *
xcodeml_getAsString(XcodeMLNode * ndPtr)
{
    char * content = NULL;

    if (ndPtr == NULL) {
        return NULL;
    }

    switch (XCODEML_TYPE(ndPtr)) {
        case XcodeML_Element: {
            content = xcodeml_GetElementValue(ndPtr);
            break;
        }

        case XcodeML_Attribute: {
            content = xcodeml_GetAttributeValue(ndPtr);
            break;
        }

        case XcodeML_Value: {
            content = XCODEML_VALUE(ndPtr);
            break;
        }

        default:
            break;
    }

    return content;
}

/**
 * Gets a child element of an attribute by Name.
 * @param ndPtr must represent element.
 * @return returns NULL if Name'd object not found.
 *        <br/> returns object if object is element.
 *        <br/> returns content if object is attribute.
 */
XcodeMLNode *
xcodeml_getByName(XcodeMLNode * ndPtr, char * name)
{
    XcodeMLNode *x;
    XcodeMLList *lp;

    if(ndPtr == NULL || name == NULL)
        return NULL;

    if (XCODEML_TYPE(ndPtr) != XcodeML_Element) {
        /* ERROR */
        /* may invalid usage. */
        return NULL;
    }

    FOR_ITEMS_IN_XCODEML_LIST(lp, ndPtr){
        x = XCODEML_LIST_NODE(lp);

        switch (XCODEML_TYPE(x)) {
            case XcodeML_Element: {
                if (strcmp(name, XCODEML_NAME(x)) == 0)
                    return x;
                break;
            }

            case XcodeML_Attribute: {
                if (strcmp(name, XCODEML_NAME(x)) == 0)
                    return XCODEML_ARG1(x);
                break;
            }

            case XcodeML_Value:
                break;

            default: {
                break;
            }
        }
    }

    return NULL;
}


/**
 * Gets a n'th child element.
 * @param ndPtr must represents an element.
 * @return returns NULL if there is no n'th child.
 *        <br/> returns n'th child.
 */
XcodeMLNode *
xcodeml_getElem(XcodeMLNode * ndPtr, int order)
{
    XcodeMLNode *x;
    XcodeMLList *lp;

    if (ndPtr == NULL || order < 0)
        return NULL;

    if (XCODEML_TYPE(ndPtr) != XcodeML_Element) {
        /* ERROR */
        /* may invalid usage. */
        return NULL;
    }

    FOR_ITEMS_IN_XCODEML_LIST(lp, ndPtr){
        x = XCODEML_LIST_NODE(lp);

        if (x == ndPtr)
            continue;

        if (XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if (order == 0)
            return x;

        order--;
    }

    return NULL;
}

static symbol_filter * filterTop = NULL;

/**
 * Pushes a new filter to the fliter stack and gets it.
 *
 * @return a new filter on the top of the filter stack.
 */
symbol_filter *
push_new_filter(void)
{
    symbol_filter * next = filterTop;
    symbol_filter * new_filter;

    new_filter = XMALLOC(symbol_filter *, sizeof(symbol_filter));

    filterTop = new_filter;
    filterTop->next = next;
    filterTop->nonUseSymbolNumber = nonUseSymbolPrefixSequence++;

    return filterTop;
}

/**
 * Gets the peek of the fliter stack.
 *
 * @return a filter on the top of the filter stack.
 */
symbol_filter *
peek_filter(void)
{
    return filterTop;
}

/**
 * Pops a filter from the fliter stack and discards it.
 */
void
pop_filter(void)
{
    if (filterTop == NULL)
        return;

    filterTop = filterTop->next;
}

/**
 * Adds the a filter element to a filter.
 *
 * @param filter a filter.
 * @param use a symbol will be used in a new environment.
 * @param local a local name of symbol.
 */
void
symbol_filter_addElem(symbol_filter * filter, char * use, char * local)
{
    symbol_filterElem * newElm = XMALLOC(symbol_filterElem *, sizeof(symbol_filterElem));

    newElm->next = filter->list;
    newElm->use = use;
    newElm->local = local;
    filter->list = newElm;
}

/**
 * Applies a filter to a symbol.
 * @param filter
 * @param symbol
 * @return NULL if symbol is NULL or a filter removed it.
 *     <br> if return value is not NULL then it is the symbol itself of a local name of it.
 */
char *
apply_filter(symbol_filter * filter, char * symbol)
{
    if (filter == NULL ||
        symbol == NULL)
        return symbol;

    if (filter->usage == RENAME) {
        symbol_filterElem * filterElem = filter->list;
        while(filterElem != NULL) {
            if (filterElem->use == NULL ||
                filterElem->local == NULL)
                break;

            if (strcmp(filterElem->use, symbol) == 0) {
                symbol =  filterElem->local;
            }

            filterElem = filterElem->next;
        }
    } else if (filter->usage == LIMIT) {

        if(xcodeml_has_symbol(symbol) == FALSE) {
            return NULL;
        }

        symbol_filterElem * filterElem = filter->list;

        while(filterElem != NULL) {
            if (filterElem->use == NULL)
                break;

            if (strcmp(filterElem->use, symbol) == 0) {
                if (filterElem->local != NULL) {
                    symbol = filterElem->local;
                }
                break;
            }

            filterElem = filterElem->next;
        }

        if (filterElem == NULL)
            symbol = NULL;
    } else if (filter->usage == ACCEPTANY) {
        return symbol;
    } else {
        fprintf(stderr, "abort in %s.", __func__);
        abort();
    }

    return apply_filter(filter->next, symbol);
}

/**
 * return if the symbol is in USE list.
 */
int
is_use_symbol(char *symbol)
{
    symbol_filterElem * filterElem;
    symbol_filter * filter;
    
    filter = peek_filter();

    if(FILTER_USAGE(filter) != LIMIT)
        return true;

    assert(filter);

    for(filterElem = filter->list; filterElem != NULL;
        filterElem = filterElem->next) {
        const char *useSym = filterElem->local ?
            filterElem->local : filterElem->use;

        if (useSym == NULL)
            break;

        if (strcmp(useSym, symbol) == 0) {
            return true;
        }
    }

    return false;
}


/**
 * get alternative symbol of which is not in use only list.
 */
char *
convert_to_non_use_symbol(char *orgname)
{
    char *symbol;
    symbol_filter * filter;
    filter = peek_filter();
    assert(filter);
    symbol = malloc(strlen(orgname) + 16);
    sprintf(symbol, "u%d_%s_", filter->nonUseSymbolNumber, orgname);
    return symbol;
}

/**
 * Gets the FfunctionDefition by the name.
 */
XcodeMLNode *
get_funcDef(XcodeMLNode * defs, char * name)
{
    XcodeMLList * lp;

    if(defs == NULL || name == NULL)
        return NULL;

    FOR_ITEMS_IN_XCODEML_LIST(lp, defs) {
        XcodeMLNode * x = XCODEML_LIST_NODE(lp);
        XcodeMLNode * nameTag;
        char * func_name;

        if(x == defs)
            continue;

        nameTag = GET_NAME(x);
        func_name = xcodeml_getAsString(nameTag);

        if(strcmp(name, func_name?:"") == 0)
            return x;
    }

    return NULL;
}
