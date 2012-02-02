#ifndef _C_TOKEN_H_
#define _C_TOKEN_H_

typedef struct {
	const char *name;
	const int code;
} CTokenInfo;

#define CTOKENINFO_SIZE 61

extern const CTokenInfo s_CTokenInfos[CTOKENINFO_SIZE];

#endif /* _C_TOKEN_H_ */

