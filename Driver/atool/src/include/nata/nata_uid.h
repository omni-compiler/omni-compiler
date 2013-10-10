/* 
 * $Id: nata_uid.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_UID_H__
#define __NATA_UID_H__


typedef uint8_t nata_Uid[32];

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

extern void		nata_initUid(void);
extern void		nata_getUid(nata_Uid *uPtr);

#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* ! __NATA_UID_H__ */
