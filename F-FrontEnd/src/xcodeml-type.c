/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "xcodeml.h"

#define XCODEML_TYPE_TABLE_SIZE 1000

xentry * type_table;

#if 0
static void
free_entry(xentry entry)
{
    entry.content = NULL;
    entry.next = NULL;
    entry.tagname = NULL;
}

void
typetable_init()
{
    int i;
    for (i = 0; i < XCODEML_TYPE_TABLE_SIZE; i++) {
        free_entry(type_table[i]);
    }
}
#endif

/**
 * Free hash entry recurcively.
 *
 * @param entry
 */
static void
free_entry(xentry * entry)
{
    if (entry == NULL)
        return;

    free_entry(entry->next);
    free(entry->next);
}

/**
 * Initialize the hash table of the types.
 */
void
typetable_init()
{
    if (type_table != NULL) {
        int i;
        for (i = 0; i < XCODEML_TYPE_TABLE_SIZE; i++) {
            free_entry(type_table + i);
        }
        free(type_table);
    }

    type_table = XMALLOC(xentry *, sizeof(xentry) * XCODEML_TYPE_TABLE_SIZE);
}

/**
 * Gets a hash value of the type signature.
 */
static int
typetable_hashcode(const char * type_signature)
{
    int hcode = 0;
    const char * ch;

    if (type_signature == NULL)
        return 0;

    ch = type_signature;

    while (*ch != '\0') {
        hcode = (hcode * CHAR_MAX) + (*ch);
        ch++;
    }

    if (hcode < 0) {
        hcode = -hcode;
    }

    hcode = (hcode % XCODEML_TYPE_TABLE_SIZE) & XCODEML_TYPE_TABLE_SIZE;

    return hcode;
}

/**
 * Inserts a type to the hash table.
 */
void
typetable_enhash(XcodeMLNode * type)
{
    xentry * entry;
    char * type_signature;
    int hcode;

    if(type == NULL)
        return;

    type_signature = GET_TYPE(type);

    if (type_signature == NULL)
        return;

    if (type_isPrimitive(type_signature) == true) {
        return;
    }

    hcode = typetable_hashcode(type_signature);

    for (entry = type_table + hcode;
         entry->content != NULL;
         entry = entry->next) {

        if(strcmp(type_signature, GET_TYPE(entry->content)) == 0)
            return;
    }

    if (entry->content == NULL) {
        entry->content = type;
        entry->next = XMALLOC(xentry *, sizeof(xentry));
    }
}

/**
 * Gets a entry of the type from the hash table by its signature.
 *
 * @param type_signature a signature of the type.
 * @return a hash entry of the type.
 *    <br>returns NULL if type_signature is NULL or
 *    <br>a type of the type_signature is not found.
 */
xentry *
typetable_dehash(char * type_signature)
{
    xentry * entry;
    int hcode;

    if(type_signature == NULL)
        return NULL;

    hcode = typetable_hashcode(type_signature);

    for (entry = type_table + hcode;
         entry != NULL && entry->content != NULL;
         entry = entry->next) {

        if (strcmp(type_signature, GET_TYPE(entry->content)) == 0) {
            return entry;
        }
    }

    return NULL;
}

/**
 * Checks if the type_signature is of a primitive type.
 *
 * @param type_signature a signature of the type.
 * @return returns true if the type is a primitive one.
 */
bool
type_isPrimitive(char * type_signature)
{
    if (type_signature == NULL)
        return false;

    if (strcmp(type_signature, "Fint") == 0 ||
        strcmp(type_signature, "Fcharacter") == 0 ||
        strcmp(type_signature, "Freal") == 0 ||
        strcmp(type_signature, "Fvoid") == 0 ||
        strcmp(type_signature, "Fnumeric") == 0 ||
        strcmp(type_signature, "FnumericAll") == 0 ||
        strcmp(type_signature, "Fcomplex") == 0 ||
        strcmp(type_signature, "Flogical") == 0) {
        return true;
    }

    return false;
}
