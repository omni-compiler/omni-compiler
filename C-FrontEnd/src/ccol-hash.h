/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file ccol-hash.h
 * hash table
 */

/*
 * From Tcl8.0.5/license.terms
 *
This software is copyrighted by the Regents of the University of
California, Sun Microsystems, Inc., Scriptics Corporation,
and other parties.  The following terms apply to all files associated
with the software unless explicitly disclaimed in individual files.

The authors hereby grant permission to use, copy, modify, distribute,
and license this software and its documentation for any purpose, provided
that existing copyright notices are retained in all copies and that this
notice is included verbatim in any distributions. No written agreement,
license, or royalty fee is required for any of the authorized uses.
Modifications to this software may be copyrighted by their authors
and need not follow the licensing terms described here, provided that
the new terms are clearly indicated on the first page of each file where
they apply.

IN NO EVENT SHALL THE AUTHORS OR DISTRIBUTORS BE LIABLE TO ANY PARTY
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
ARISING OUT OF THE USE OF THIS SOFTWARE, ITS DOCUMENTATION, OR ANY
DERIVATIVES THEREOF, EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

THE AUTHORS AND DISTRIBUTORS SPECIFICALLY DISCLAIM ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.  THIS SOFTWARE
IS PROVIDED ON AN "AS IS" BASIS, AND THE AUTHORS AND DISTRIBUTORS HAVE
NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.

GOVERNMENT USE: If you are acquiring this software on behalf of the
U.S. government, the Government shall have only "Restricted Rights"
in the software and related documentation as defined in the Federal 
Acquisition Regulations (FARs) in Clause 52.227.19 (c) (2).  If you
are acquiring the software on behalf of the Department of Defense, the
software shall be classified as "Commercial Computer Software" and the
Government shall have only "Restricted Rights" as defined in Clause
252.227-7013 (c) (1) of DFARs.  Notwithstanding the foregoing, the
authors grant the U.S. Government and others acting in its behalf
permission to use and distribute the software in accordance with the
terms specified in this license. 
*/

#ifndef _CCOL_HASH_H_
#define _CCOL_HASH_H_

#include "ccol-cmn.h"

typedef CCOL_Data ClientData;

/**
 * Forward declaration of CCOL_HashTable.  Needed by some C++ compilers
 * to prevent errors when the forward reference to CCOL_HashTable is
 * encountered in the CCOL_HashEntry structure.
 */

#ifdef __cplusplus
struct CCOL_HashTable;
#endif

/**
 * \brief
 * hash entry
 *
 * Structure definition for an entry in a hash table.  No-one outside
 * ns should access any of these fields directly;  use the macros
 * defined below.
 */

typedef struct CCOL_HashEntry {
    struct CCOL_HashEntry *nextPtr;    /** Pointer to next entry in this
                                        * hash bucket, or NULL for end of
                                        * chain. */
    struct CCOL_HashTable *tablePtr;   /** Pointer to table containing entry. */
    struct CCOL_HashEntry **bucketPtr; /** Pointer to bucket that points to
                                        * first entry in this entry's chain:
                                        * used for deleting the entry. */
    ClientData clientData;             /** Application stores something here
                                        * with ccol_SetHashValue. */
    union {                            /** Key has one of these forms: */
        const char* oneWordValue;      /** One-word value for key. */
        int words[1];                  /** Multiple integer words for key.
                                        * The actual size will be as large
                                        * as necessary for this table's
                                        * keys. */
        char string[4];                /** String for key.  The actual size
                                        * will be as large as needed to hold
                                        * the key. */
    } key;                             /** MUST BE LAST FIELD IN RECORD!! */
} CCOL_HashEntry;

#define CCOL_HT_SMALL_HASH_TABLE 4
/**
 * \brief
 * hash table
 *
 * Structure definition for a hash table.  Must be in ns.h so clients
 * can allocate space for these structures, but clients should never
 * access any fields in this structure.
 */

typedef struct CCOL_HashTable {
    CCOL_HashEntry **buckets;        /** Pointer to bucket array.  Each
                                      * element points to first entry in
                                      * bucket's hash chain, or NULL. */
    CCOL_HashEntry *staticBuckets[CCOL_HT_SMALL_HASH_TABLE];
                                     /** Bucket array used for small tables
                                      * (to avoid mallocs and frees). */
    int numBuckets;                  /** Total number of buckets allocated
                                      * at **bucketPtr. */
    int numEntries;                  /** Total number of entries present
                                      * in table. */
    int rebuildSize;                 /** Enlarge table when numEntries gets
                                      * to be this large. */
    int downShift;                   /** Shift count used in hashing
                                      * function.  Designed to use high-
                                      * order bits of randomized keys. */
    int mask;                        /** Mask value used in hashing
                                      * function. */
    int keyType;                     /** Type of keys used in this table. 
                                      * It's either CCOL_HT_STRING_KEYS,
                                      * CCOL_HT_ONE_WORD_KEYS, or an integer
                                      * giving the number of ints that
                                      * is the size of the key.
                                      */
    CCOL_HashEntry *(*findProc) (struct CCOL_HashTable *tablePtr,
        const char *key);
    CCOL_HashEntry *(*createProc) (struct CCOL_HashTable *tablePtr,
        const char *key, int *newPtr);
} CCOL_HashTable;

/**
 * Structure definition for information used to keep track of searches
 * through hash tables:
 */

typedef struct CCOL_HashSearch {
    CCOL_HashTable *tablePtr;        /** Table being searched. */
    int nextIndex;                   /** Index of next bucket to be
                                      * enumerated after present one. */
    CCOL_HashEntry *nextEntryPtr;    /** Next entry to be enumerated in the
                                      * the current bucket. */
} CCOL_HashSearch;

/**
 * Acceptable key types for hash tables:
 */

#define CCOL_HT_STRING_KEYS      0
#define CCOL_HT_ONE_WORD_KEYS    1

/**
 * Macros for clients to use to access fields of hash entries:
 */

#define ccol_GetHashValue(h) ((h)->clientData)
#define ccol_SetHashValue(h, value) ((h)->clientData = (ClientData) (value))
#define ccol_GetHashKey(tablePtr, h) \
    ((char *) (((tablePtr)->keyType == CCOL_HT_ONE_WORD_KEYS) ? (h)->key.oneWordValue \
                        : (h)->key.string))

/**
 * Macros to use for clients to use to invoke find and create procedures
 * for hash tables:
 */

#define ccol_FindHashEntry(tablePtr, key) \
    (*((tablePtr)->findProc))(tablePtr, key)
#define ccol_CreateHashEntry(tablePtr, key, newPtr) \
    (*((tablePtr)->createProc))(tablePtr, key, newPtr)

extern void            ccol_DeleteHashEntry(CCOL_HashEntry *entryPtr);
extern void            ccol_DeleteHashTable(CCOL_HashTable *tablePtr);
extern CCOL_HashEntry* ccol_FirstHashEntry(CCOL_HashTable *tablePtr,
                       CCOL_HashSearch *searchPtr);
extern char*           ccol_HashStats(CCOL_HashTable *tablePtr);
extern void            ccol_InitHashTable(CCOL_HashTable *tablePtr,
                       int keyType);
extern CCOL_HashEntry* ccol_NextHashEntry (CCOL_HashSearch *searchPtr);

/*
 * Extension
 */

extern CCOL_HashEntry* ccol_PutStringHashEntry(CCOL_HashTable *tablePtr, const char *key, ClientData d);
extern ClientData      ccol_RemoveStringHashEntry(CCOL_HashTable *tablePtr, const char *key);
extern void            ccol_ClearHashTable(CCOL_HashTable *tablePtr);

#define CCOL_HT_SIZE(tablePtr)              ((tablePtr)->numEntries)
#define CCOL_HT_DATA(h)                     ccol_GetHashValue(h)
#define CCOL_HT_SET_DATA(h, d)              ccol_SetHashValue(h, d)
#define CCOL_HT_KEY(tablePtr, h)            ccol_GetHashKey(tablePtr, h)
#define CCOL_HT_FIND_STR(tablePtr, k)       ccol_FindHashEntry(tablePtr, k)
#define CCOL_HT_FIND_WORD(tablePtr, k)      ccol_FindHashEntry(tablePtr, (const char*)k)
#define CCOL_HT_PUT_STR(tablePtr, k, d)     ccol_PutStringHashEntry(tablePtr, k, d)
#define CCOL_HT_PUT_WORD(tablePtr, k, d)    ccol_PutStringHashEntry(tablePtr, (const char*)(k), d)
#define CCOL_HT_REMOVE_STR(tablePtr, k)     ccol_RemoveStringHashEntry(tablePtr, k)
#define CCOL_HT_REMOVE_WORD(tablePtr, k)    ccol_RemoveStringHashEntry(tablePtr, (const char*)(k))
#define CCOL_HT_INIT(tablePtr, type)        ccol_InitHashTable(tablePtr, type)
#define CCOL_HT_CLEAR(tablePtr)             ccol_ClearHashTable(tablePtr)
#define CCOL_HT_DESTROY(tablePtr)           ccol_DeleteHashTable(tablePtr)

#define CCOL_HT_FOREACH(entryPtr, search, tablePtr) \
        for(entryPtr = ccol_FirstHashEntry((tablePtr), &(search));\
            entryPtr != NULL; entryPtr = ccol_NextHashEntry(&(search)))

#endif /* _CCOL_HAHS_H */
