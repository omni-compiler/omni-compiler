/*
 * $Id: nata_rcsid.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_RCSID_H__
#define __NATA_RCSID_H__

#ifdef __GNUC__
#define USE_BOODOO	__attribute__((used))
#else
#define USE_BOODOO	/**/
#endif /* __GNUC__ */

#define __rcsId(id) \
static volatile const char USE_BOODOO *rcsid(void) { \
    return id ? id : rcsid(); \
}

#endif /* __NATA_RCSID_H__ */

