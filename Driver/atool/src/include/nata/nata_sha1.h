/* 
 * $Id: nata_sha1.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_SHA1_H__
#define __NATA_SHA1_H__

typedef struct {
    uint32_t state[5];
    uint32_t count[2];
    uint8_t buffer[64];
} nata_SHA1_Context;

typedef uint8_t nata_SHA1_Digest[20];

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

extern void	nata_initSha1(nata_SHA1_Context *cPtr);
extern void	nata_updateSha1(nata_SHA1_Context *cPtr,
                                const uint8_t *data,
                                size_t len);
extern void	nata_finalSha1(nata_SHA1_Digest digest,
                               nata_SHA1_Context *cPtr);

extern void	nata_getSha1(const uint8_t *data, size_t len,
                             nata_SHA1_Digest digest);

#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* ! __NATA_SHA1_H__ */
