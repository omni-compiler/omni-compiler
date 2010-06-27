/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ccol-hash.h"

/*
 * When there are this many entries per bucket, on average, rebuild
 * the hash table to make it larger.
 */

#define REBUILD_MULTIPLIER        3


/*
 * The following macro takes a preliminary integer hash value and
 * produces an index into a hash tables bucket list.  The idea is
 * to make it so that preliminary values that are arbitrarily similar
 * will end up in different buckets.  The hash function was taken
 * from a random-number generator.
 */

#define RANDOM_INDEX(tablePtr, i) \
    (((((long) (i))*1103515245) >> (tablePtr)->downShift) & (tablePtr)->mask)

/*
 * Procedure prototypes for static procedures in this file:
 */

static CCOL_HashEntry *    ArrayFind (CCOL_HashTable *tablePtr, const char *key);
static CCOL_HashEntry *    ArrayCreate (CCOL_HashTable *tablePtr, const char *key, int *newPtr);
static CCOL_HashEntry *    BogusFind (CCOL_HashTable *tablePtr, const char *key);
static CCOL_HashEntry *    BogusCreate (CCOL_HashTable *tablePtr, const char *key, int *newPtr);
static unsigned int            HashString (const char *string);
static void                    RebuildTable (CCOL_HashTable *tablePtr);
static CCOL_HashEntry *    StringFind (CCOL_HashTable *tablePtr, const char *key);
static CCOL_HashEntry *    StringCreate (CCOL_HashTable *tablePtr, const char *key, int *newPtr);
static CCOL_HashEntry *    OneWordFind (CCOL_HashTable *tablePtr, const char *key);
static CCOL_HashEntry *    OneWordCreate (CCOL_HashTable *tablePtr, const char *key, int *newPtr);

/*
 *----------------------------------------------------------------------
 *
 * ccol_InitHashTable --
 *
 *        Given storage for a hash table, set up the fields to prepare
 *        the hash table for use.
 *
 * Results:
 *        None.
 *
 * Side effects:
 *        TablePtr is now ready to be passed to ccol_FindHashEntry and
 *        ccol_CreateHashEntry.
 *
 *----------------------------------------------------------------------
 */

void
ccol_InitHashTable(tablePtr, keyType)
    register CCOL_HashTable *tablePtr;        /* Pointer to table record, which
                                         * is supplied by the caller. */
    int keyType;                        /* Type of keys to use in table:
                                         * CCOL_HT_STRING_KEYS, CCOL_HT_ONE_WORD_KEYS,
                                         * or an integer >= 2. */
{
    tablePtr->buckets = tablePtr->staticBuckets;
    tablePtr->staticBuckets[0] = tablePtr->staticBuckets[1] = 0;
    tablePtr->staticBuckets[2] = tablePtr->staticBuckets[3] = 0;
    tablePtr->numBuckets = CCOL_HT_SMALL_HASH_TABLE;
    tablePtr->numEntries = 0;
    tablePtr->rebuildSize = CCOL_HT_SMALL_HASH_TABLE*REBUILD_MULTIPLIER;
    tablePtr->downShift = 28;
    tablePtr->mask = 3;
    tablePtr->keyType = keyType;
    if (keyType == CCOL_HT_STRING_KEYS) {
        tablePtr->findProc = StringFind;
        tablePtr->createProc = StringCreate;
    } else if (keyType == CCOL_HT_ONE_WORD_KEYS) {
        tablePtr->findProc = OneWordFind;
        tablePtr->createProc = OneWordCreate;
    } else {
        tablePtr->findProc = ArrayFind;
        tablePtr->createProc = ArrayCreate;
    };
}

/*
 *----------------------------------------------------------------------
 *
 * ccol_DeleteHashEntry --
 *
 *        Remove a single entry from a hash table.
 *
 * Results:
 *        None.
 *
 * Side effects:
 *        The entry given by entryPtr is deleted from its table and
 *        should never again be used by the caller.  It is up to the
 *        caller to free the clientData field of the entry, if that
 *        is relevant.
 *
 *----------------------------------------------------------------------
 */

void
ccol_DeleteHashEntry(entryPtr)
    CCOL_HashEntry *entryPtr;
{
    register CCOL_HashEntry *prevPtr;

    if (*entryPtr->bucketPtr == entryPtr) {
        *entryPtr->bucketPtr = entryPtr->nextPtr;
    } else {
        for (prevPtr = *entryPtr->bucketPtr; ; prevPtr = prevPtr->nextPtr) {
            if (prevPtr == NULL) {
                fprintf(stderr, "malformed bucket chain in ccol_DeleteHashEntry.\n");
                abort();
            }
            if (prevPtr->nextPtr == entryPtr) {
                prevPtr->nextPtr = entryPtr->nextPtr;
                break;
            }
        }
    }
    entryPtr->tablePtr->numEntries--;
    ccol_Free((char *) entryPtr);
}

/*
 *----------------------------------------------------------------------
 *
 * ccol_DeleteHashTable --
 *
 *        Free up everything associated with a hash table except for
 *        the record for the table itself.
 *
 * Results:
 *        None.
 *
 * Side effects:
 *        The hash table is no longer useable.
 *
 *----------------------------------------------------------------------
 */

void
ccol_DeleteHashTable(tablePtr)
    register CCOL_HashTable *tablePtr;                /* Table to delete. */
{
    register CCOL_HashEntry *hPtr, *nextPtr;
    int i;

    /*
     * Free up all the entries in the table.
     */

    for (i = 0; i < tablePtr->numBuckets; i++) {
        hPtr = tablePtr->buckets[i];
        while (hPtr != NULL) {
            nextPtr = hPtr->nextPtr;
            ccol_Free((char *) hPtr);
            hPtr = nextPtr;
        }
    }

    /*
     * Free up the bucket array, if it was dynamically allocated.
     */

    if (tablePtr->buckets != tablePtr->staticBuckets) {
        ccol_Free((char *) tablePtr->buckets);
    }

    /*
     * Arrange for panics if the table is used again without
     * re-initialization.
     */

    tablePtr->findProc = BogusFind;
    tablePtr->createProc = BogusCreate;
}

/*
 *----------------------------------------------------------------------
 *
 * ccol_FirstHashEntry --
 *
 *        Locate the first entry in a hash table and set up a record
 *        that can be used to step through all the remaining entries
 *        of the table.
 *
 * Results:
 *        The return value is a pointer to the first entry in tablePtr,
 *        or NULL if tablePtr has no entries in it.  The memory at
 *        *searchPtr is initialized so that subsequent calls to
 *        ccol_NextHashEntry will return all of the entries in the table,
 *        one at a time.
 *
 * Side effects:
 *        None.
 *
 *----------------------------------------------------------------------
 */

CCOL_HashEntry *
ccol_FirstHashEntry(tablePtr, searchPtr)
    CCOL_HashTable *tablePtr;                /* Table to search. */
    CCOL_HashSearch *searchPtr;        /* Place to store information about
                                         * progress through the table. */
{
    searchPtr->tablePtr = tablePtr;
    searchPtr->nextIndex = 0;
    searchPtr->nextEntryPtr = NULL;
    return ccol_NextHashEntry(searchPtr);
}

/*
 *----------------------------------------------------------------------
 *
 * ccol_NextHashEntry --
 *
 *        Once a hash table enumeration has been initiated by calling
 *        ccol_FirstHashEntry, this procedure may be called to return
 *        successive elements of the table.
 *
 * Results:
 *        The return value is the next entry in the hash table being
 *        enumerated, or NULL if the end of the table is reached.
 *
 * Side effects:
 *        None.
 *
 *----------------------------------------------------------------------
 */

CCOL_HashEntry *
ccol_NextHashEntry(searchPtr)
    register CCOL_HashSearch *searchPtr;        /* Place to store information about
                                                 * progress through the table.  Must
                                                 * have been initialized by calling
                                                 * ccol_FirstHashEntry. */
{
    CCOL_HashEntry *hPtr;

    while (searchPtr->nextEntryPtr == NULL) {
        if (searchPtr->nextIndex >= searchPtr->tablePtr->numBuckets) {
            return NULL;
        }
        searchPtr->nextEntryPtr =
                searchPtr->tablePtr->buckets[searchPtr->nextIndex];
        searchPtr->nextIndex++;
    }
    hPtr = searchPtr->nextEntryPtr;
    searchPtr->nextEntryPtr = hPtr->nextPtr;
    return hPtr;
}

/*
 *----------------------------------------------------------------------
 *
 * ccol_HashStats --
 *
 *        Return statistics describing the layout of the hash table
 *        in its hash buckets.
 *
 * Results:
 *        The return value is a malloc-ed string containing information
 *        about tablePtr.  It is the caller's responsibility to free
 *        this string.
 *
 * Side effects:
 *        None.
 *
 *----------------------------------------------------------------------
 */

char *
ccol_HashStats(tablePtr)
    CCOL_HashTable *tablePtr;        /* Table for which to produce stats. */
{
#define NUM_COUNTERS 10
    int count[NUM_COUNTERS], overflow, i, j;
    double average, tmp;
    register CCOL_HashEntry *hPtr;
    char *result, *p;

    /*
     * Compute a histogram of bucket usage.
     */

    for (i = 0; i < NUM_COUNTERS; i++) {
        count[i] = 0;
    }
    overflow = 0;
    average = 0.0;
    for (i = 0; i < tablePtr->numBuckets; i++) {
        j = 0;
        for (hPtr = tablePtr->buckets[i]; hPtr != NULL; hPtr = hPtr->nextPtr) {
            j++;
        }
        if (j < NUM_COUNTERS) {
            count[j]++;
        } else {
            overflow++;
        }
        tmp = j;
        average += (tmp+1.0)*(tmp/tablePtr->numEntries)/2.0;
    }

    /*
     * Print out the histogram and a few other pieces of information.
     */

    result = (char *)ccol_Malloc((unsigned) ((NUM_COUNTERS*60) + 300));
    sprintf(result, "%d entries in table, %d buckets\n",
            tablePtr->numEntries, tablePtr->numBuckets);
    p = result + strlen(result);
    for (i = 0; i < NUM_COUNTERS; i++) {
        sprintf(p, "number of buckets with %d entries: %d\n",
                i, count[i]);
        p += strlen(p);
    }
    sprintf(p, "number of buckets with %d or more entries: %d\n",
            NUM_COUNTERS, overflow);
    p += strlen(p);
    sprintf(p, "average search distance for entry: %.1f", average);
    return result;
}

/*
 *----------------------------------------------------------------------
 *
 * HashString --
 *
 *        Compute a one-word summary of a text string, which can be
 *        used to generate a hash index.
 *
 * Results:
 *        The return value is a one-word summary of the information in
 *        string.
 *
 * Side effects:
 *        None.
 *
 *----------------------------------------------------------------------
 */

static unsigned int
HashString(string)
    register const char *string;/* String from which to compute hash value. */
{
    register unsigned int result;
    register int c;

    /*
     * I tried a zillion different hash functions and asked many other
     * people for advice.  Many people had their own favorite functions,
     * all different, but no-one had much idea why they were good ones.
     * I chose the one below (multiply by 9 and add new character)
     * because of the following reasons:
     *
     * 1. Multiplying by 10 is perfect for keys that are decimal strings,
     *    and multiplying by 9 is just about as good.
     * 2. Times-9 is (shift-left-3) plus (old).  This means that each
     *    character's bits hang around in the low-order bits of the
     *    hash value for ever, plus they spread fairly rapidly up to
     *    the high-order bits to fill out the hash value.  This seems
     *    works well both for decimal and non-decimal strings.
     */

    result = 0;
    while (1) {
        c = *string;
        string++;
        if (c == 0) {
            break;
        }
        result += (result<<3) + c;
    }
    return result;
}

/*
 *----------------------------------------------------------------------
 *
 * StringFind --
 *
 *        Given a hash table with string keys, and a string key, find
 *        the entry with a matching key.
 *
 * Results:
 *        The return value is a token for the matching entry in the
 *        hash table, or NULL if there was no matching entry.
 *
 * Side effects:
 *        None.
 *
 *----------------------------------------------------------------------
 */

static CCOL_HashEntry *
StringFind(tablePtr, key)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    const char *key;                /* Key to use to find matching entry. */
{
    register CCOL_HashEntry *hPtr;
    register const char *p1, *p2;
    int index;

    index = HashString(key) & tablePtr->mask;

    /*
     * Search all of the entries in the appropriate bucket.
     */

    for (hPtr = tablePtr->buckets[index]; hPtr != NULL;
            hPtr = hPtr->nextPtr) {
        for (p1 = key, p2 = hPtr->key.string; ; p1++, p2++) {
            if (*p1 != *p2) {
                break;
            }
            if (*p1 == '\0') {
                return hPtr;
            }
        }
    }
    return NULL;
}

/*
 *----------------------------------------------------------------------
 *
 * StringCreate --
 *
 *        Given a hash table with string keys, and a string key, find
 *        the entry with a matching key.  If there is no matching entry,
 *        then create a new entry that does match.
 *
 * Results:
 *        The return value is a pointer to the matching entry.  If this
 *        is a newly-created entry, then *newPtr will be set to a non-zero
 *        value;  otherwise *newPtr will be set to 0.  If this is a new
 *        entry the value stored in the entry will initially be 0.
 *
 * Side effects:
 *        A new entry may be added to the hash table.
 *
 *----------------------------------------------------------------------
 */

static CCOL_HashEntry *
StringCreate(tablePtr, key, newPtr)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    const char *key;                /* Key to use to find or create matching
                                 * entry. */
    int *newPtr;                /* Store info here telling whether a new
                                 * entry was created. */
{
    register CCOL_HashEntry *hPtr;
    register const char *p1, *p2;
    int index;

    index = HashString(key) & tablePtr->mask;

    /*
     * Search all of the entries in this bucket.
     */

    for (hPtr = tablePtr->buckets[index]; hPtr != NULL;
            hPtr = hPtr->nextPtr) {
        for (p1 = key, p2 = hPtr->key.string; ; p1++, p2++) {
            if (*p1 != *p2) {
                break;
            }
            if (*p1 == '\0') {
                *newPtr = 0;
                return hPtr;
            }
        }
    }

    /*
     * Entry not found.  Add a new one to the bucket.
     */

    *newPtr = 1;
    hPtr = (CCOL_HashEntry *)ccol_Malloc((unsigned)
            (sizeof(CCOL_HashEntry) + strlen(key) - (sizeof(hPtr->key) -1)));
    hPtr->tablePtr = tablePtr;
    hPtr->bucketPtr = &(tablePtr->buckets[index]);
    hPtr->nextPtr = *hPtr->bucketPtr;
    hPtr->clientData = 0;
    strcpy(hPtr->key.string, key);
    *hPtr->bucketPtr = hPtr;
    tablePtr->numEntries++;

    /*
     * If the table has exceeded a decent size, rebuild it with many
     * more buckets.
     */

    if (tablePtr->numEntries >= tablePtr->rebuildSize) {
        RebuildTable(tablePtr);
    }
    return hPtr;
}

/*
 *----------------------------------------------------------------------
 *
 * OneWordFind --
 *
 *        Given a hash table with one-word keys, and a one-word key, find
 *        the entry with a matching key.
 *
 * Results:
 *        The return value is a token for the matching entry in the
 *        hash table, or NULL if there was no matching entry.
 *
 * Side effects:
 *        None.
 *
 *----------------------------------------------------------------------
 */

static CCOL_HashEntry *
OneWordFind(tablePtr, key)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    register const char *key;        /* Key to use to find matching entry. */
{
    register CCOL_HashEntry *hPtr;
    int index;

    index = RANDOM_INDEX(tablePtr, key);

    /*
     * Search all of the entries in the appropriate bucket.
     */

    for (hPtr = tablePtr->buckets[index]; hPtr != NULL;
            hPtr = hPtr->nextPtr) {
        if (hPtr->key.oneWordValue == key) {
            return hPtr;
        }
    }
    return NULL;
}

/*
 *----------------------------------------------------------------------
 *
 * OneWordCreate --
 *
 *        Given a hash table with one-word keys, and a one-word key, find
 *        the entry with a matching key.  If there is no matching entry,
 *        then create a new entry that does match.
 *
 * Results:
 *        The return value is a pointer to the matching entry.  If this
 *        is a newly-created entry, then *newPtr will be set to a non-zero
 *        value;  otherwise *newPtr will be set to 0.  If this is a new
 *        entry the value stored in the entry will initially be 0.
 *
 * Side effects:
 *        A new entry may be added to the hash table.
 *
 *----------------------------------------------------------------------
 */

static CCOL_HashEntry *
OneWordCreate(tablePtr, key, newPtr)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    register const char *key;        /* Key to use to find or create matching
                                 * entry. */
    int *newPtr;                /* Store info here telling whether a new
                                 * entry was created. */
{
    register CCOL_HashEntry *hPtr;
    int index;

    index = RANDOM_INDEX(tablePtr, key);

    /*
     * Search all of the entries in this bucket.
     */

    for (hPtr = tablePtr->buckets[index]; hPtr != NULL;
            hPtr = hPtr->nextPtr) {
        if (hPtr->key.oneWordValue == key) {
            *newPtr = 0;
            return hPtr;
        }
    }

    /*
     * Entry not found.  Add a new one to the bucket.
     */

    *newPtr = 1;
    hPtr = (CCOL_HashEntry *)ccol_Malloc(sizeof(CCOL_HashEntry));
    hPtr->tablePtr = tablePtr;
    hPtr->bucketPtr = &(tablePtr->buckets[index]);
    hPtr->nextPtr = *hPtr->bucketPtr;
    hPtr->clientData = 0;
    hPtr->key.oneWordValue = key;
    *hPtr->bucketPtr = hPtr;
    tablePtr->numEntries++;

    /*
     * If the table has exceeded a decent size, rebuild it with many
     * more buckets.
     */

    if (tablePtr->numEntries >= tablePtr->rebuildSize) {
        RebuildTable(tablePtr);
    }
    return hPtr;
}

/*
 *----------------------------------------------------------------------
 *
 * ArrayFind --
 *
 *        Given a hash table with array-of-int keys, and a key, find
 *        the entry with a matching key.
 *
 * Results:
 *        The return value is a token for the matching entry in the
 *        hash table, or NULL if there was no matching entry.
 *
 * Side effects:
 *        None.
 *
 *----------------------------------------------------------------------
 */

static CCOL_HashEntry *
ArrayFind(tablePtr, key)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    const char *key;                /* Key to use to find matching entry. */
{
    register CCOL_HashEntry *hPtr;
    int *arrayPtr = (int *) key;
    register int *iPtr1, *iPtr2;
    int index, count;

    for (index = 0, count = tablePtr->keyType, iPtr1 = arrayPtr;
            count > 0; count--, iPtr1++) {
        index += *iPtr1;
    }
    index = RANDOM_INDEX(tablePtr, index);

    /*
     * Search all of the entries in the appropriate bucket.
     */

    for (hPtr = tablePtr->buckets[index]; hPtr != NULL;
            hPtr = hPtr->nextPtr) {
        for (iPtr1 = arrayPtr, iPtr2 = hPtr->key.words,
                count = tablePtr->keyType; ; count--, iPtr1++, iPtr2++) {
            if (count == 0) {
                return hPtr;
            }
            if (*iPtr1 != *iPtr2) {
                break;
            }
        }
    }
    return NULL;
}

/*
 *----------------------------------------------------------------------
 *
 * ArrayCreate --
 *
 *        Given a hash table with one-word keys, and a one-word key, find
 *        the entry with a matching key.  If there is no matching entry,
 *        then create a new entry that does match.
 *
 * Results:
 *        The return value is a pointer to the matching entry.  If this
 *        is a newly-created entry, then *newPtr will be set to a non-zero
 *        value;  otherwise *newPtr will be set to 0.  If this is a new
 *        entry the value stored in the entry will initially be 0.
 *
 * Side effects:
 *        A new entry may be added to the hash table.
 *
 *----------------------------------------------------------------------
 */

static CCOL_HashEntry *
ArrayCreate(tablePtr, key, newPtr)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    register const char *key;        /* Key to use to find or create matching
                                 * entry. */
    int *newPtr;                /* Store info here telling whether a new
                                 * entry was created. */
{
    register CCOL_HashEntry *hPtr;
    int *arrayPtr = (int *) key;
    register int *iPtr1, *iPtr2;
    int index, count;

    for (index = 0, count = tablePtr->keyType, iPtr1 = arrayPtr;
            count > 0; count--, iPtr1++) {
        index += *iPtr1;
    }
    index = RANDOM_INDEX(tablePtr, index);

    /*
     * Search all of the entries in the appropriate bucket.
     */

    for (hPtr = tablePtr->buckets[index]; hPtr != NULL;
            hPtr = hPtr->nextPtr) {
        for (iPtr1 = arrayPtr, iPtr2 = hPtr->key.words,
                count = tablePtr->keyType; ; count--, iPtr1++, iPtr2++) {
            if (count == 0) {
                *newPtr = 0;
                return hPtr;
            }
            if (*iPtr1 != *iPtr2) {
                break;
            }
        }
    }

    /*
     * Entry not found.  Add a new one to the bucket.
     */

    *newPtr = 1;
    hPtr = (CCOL_HashEntry *)ccol_Malloc((unsigned) (sizeof(CCOL_HashEntry)
            + (tablePtr->keyType*sizeof(int)) - 4));
    hPtr->tablePtr = tablePtr;
    hPtr->bucketPtr = &(tablePtr->buckets[index]);
    hPtr->nextPtr = *hPtr->bucketPtr;
    hPtr->clientData = 0;
    for (iPtr1 = arrayPtr, iPtr2 = hPtr->key.words, count = tablePtr->keyType;
            count > 0; count--, iPtr1++, iPtr2++) {
        *iPtr2 = *iPtr1;
    }
    *hPtr->bucketPtr = hPtr;
    tablePtr->numEntries++;

    /*
     * If the table has exceeded a decent size, rebuild it with many
     * more buckets.
     */

    if (tablePtr->numEntries >= tablePtr->rebuildSize) {
        RebuildTable(tablePtr);
    }
    return hPtr;
}

/*
 *----------------------------------------------------------------------
 *
 * BogusFind --
 *
 *        This procedure is invoked when an ccol_FindHashEntry is called
 *        on a table that has been deleted.
 *
 * Results:
 *        If panic returns (which it shouldn't) this procedure returns
 *        NULL.
 *
 * Side effects:
 *        Generates a panic.
 *
 *----------------------------------------------------------------------
 */

        /* ARGSUSED */
static CCOL_HashEntry *
BogusFind(tablePtr, key)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    const char *key;                /* Key to use to find matching entry. */
{
    fprintf(stderr, "called ccol_FindHashEntry on deleted table.\n");
    abort();
    return NULL;
}

/*
 *----------------------------------------------------------------------
 *
 * BogusCreate --
 *
 *        This procedure is invoked when an ccol_CreateHashEntry is called
 *        on a table that has been deleted.
 *
 * Results:
 *        If panic returns (which it shouldn't) this procedure returns
 *        NULL.
 *
 * Side effects:
 *        Generates a panic.
 *
 *----------------------------------------------------------------------
 */

        /* ARGSUSED */
static CCOL_HashEntry *
BogusCreate(tablePtr, key, newPtr)
    CCOL_HashTable *tablePtr;        /* Table in which to lookup entry. */
    const char *key;                /* Key to use to find or create matching
                                 * entry. */
    int *newPtr;                /* Store info here telling whether a new
                                 * entry was created. */
{
    fprintf(stderr, "called ccol_CreateHashEntry on deleted table.\n");
    abort();
    return NULL;
}

/*
 *----------------------------------------------------------------------
 *
 * RebuildTable --
 *
 *        This procedure is invoked when the ratio of entries to hash
 *        buckets becomes too large.  It creates a new table with a
 *        larger bucket array and moves all of the entries into the
 *        new table.
 *
 * Results:
 *        None.
 *
 * Side effects:
 *        Memory gets reallocated and entries get re-hashed to new
 *        buckets.
 *
 *----------------------------------------------------------------------
 */

static void
RebuildTable(tablePtr)
    register CCOL_HashTable *tablePtr;        /* Table to enlarge. */
{
    int oldSize, count, index;
    CCOL_HashEntry **oldBuckets;
    register CCOL_HashEntry **oldChainPtr, **newChainPtr;
    register CCOL_HashEntry *hPtr;

    oldSize = tablePtr->numBuckets;
    oldBuckets = tablePtr->buckets;

    /*
     * Allocate and initialize the new bucket array, and set up
     * hashing constants for new array size.
     */

    tablePtr->numBuckets *= 4;
    tablePtr->buckets = (CCOL_HashEntry **)ccol_Malloc((unsigned)
            (tablePtr->numBuckets * sizeof(CCOL_HashEntry *)));
    for (count = tablePtr->numBuckets, newChainPtr = tablePtr->buckets;
            count > 0; count--, newChainPtr++) {
        *newChainPtr = NULL;
    }
    tablePtr->rebuildSize *= 4;
    tablePtr->downShift -= 2;
    tablePtr->mask = (tablePtr->mask << 2) + 3;

    /*
     * Rehash all of the existing entries into the new bucket array.
     */

    for (oldChainPtr = oldBuckets; oldSize > 0; oldSize--, oldChainPtr++) {
        for (hPtr = *oldChainPtr; hPtr != NULL; hPtr = *oldChainPtr) {
            *oldChainPtr = hPtr->nextPtr;
            if (tablePtr->keyType == CCOL_HT_STRING_KEYS) {
                index = HashString(hPtr->key.string) & tablePtr->mask;
            } else if (tablePtr->keyType == CCOL_HT_ONE_WORD_KEYS) {
                index = RANDOM_INDEX(tablePtr, hPtr->key.oneWordValue);
            } else {
                register int *iPtr;
                int count;

                for (index = 0, count = tablePtr->keyType,
                        iPtr = hPtr->key.words; count > 0; count--, iPtr++) {
                    index += *iPtr;
                }
                index = RANDOM_INDEX(tablePtr, index);
            }
            hPtr->bucketPtr = &(tablePtr->buckets[index]);
            hPtr->nextPtr = *hPtr->bucketPtr;
            *hPtr->bucketPtr = hPtr;
        }
    }

    /*
     * Free up the old bucket array, if it was dynamically allocated.
     */

    if (oldBuckets != tablePtr->staticBuckets) {
        ccol_Free((char *) oldBuckets);
    }
}

CCOL_HashEntry*
ccol_PutStringHashEntry(CCOL_HashTable *tablePtr, const char *key, ClientData d)
{
    int isNew;
    CCOL_HashEntry *e = ccol_CreateHashEntry(tablePtr, key, &isNew);
    CCOL_HT_SET_DATA(e, d);

    return e;
}

ClientData
ccol_RemoveStringHashEntry(CCOL_HashTable *tablePtr, const char *key)
{
    CCOL_HashEntry *e = ccol_FindHashEntry(tablePtr, key);
    CCOL_Data d;

    if(e == NULL)
        return NULL;

    d = CCOL_HT_DATA(e);
    ccol_DeleteHashEntry(e);

    return d;
}

void
ccol_ClearHashTable(CCOL_HashTable *tablePtr)
{
    int keyType = tablePtr->keyType;
    ccol_DeleteHashTable(tablePtr);
    ccol_InitHashTable(tablePtr, keyType);
}

