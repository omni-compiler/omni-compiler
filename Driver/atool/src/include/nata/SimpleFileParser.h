/* 
 * $Id: SimpleFileParser.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __SIMPLEFILEPARSER_H__
#define __SIMPLEFILEPARSER_H__

#include <nata/nata_rcsid.h>
#include <nata/nata_includes.h>
#include <nata/nata_util.h>
#include <nata/nata_macros.h>

#include <nata/SynchronizedMap.h>
#include <nata/ScopedLock.h>
#include <string>

#include <nata/nata_perror.h>





class SimpleFileParser {


private:
    __rcsId("$Id: SimpleFileParser.h 86 2012-07-30 05:33:07Z m-hirano $")


    typedef bool (*ParseProcT)(const char *path, int lineNo,
                               const char *line,
                               int argc, char * const argv[], void *ctx);





    class LineParser {


    private:
        const char *mKey;
        ParseProcT mProc;
        void *mCtx;


        LineParser(const LineParser &obj);
        LineParser operator = (const LineParser &obj);


    public:
        LineParser(const char *key, ParseProcT p, void *ctx) :
            mKey(NULL),
            mProc(p),
            mCtx(ctx) {

            if (isValidString(key) == true) {
                mKey = (const char *)strdup(key);
                if (mKey == NULL) {
                    fatal("Can't allocate a key.\n");
                }
            } else {
                fatal("The key is not a valid string.\n");
            }
        }


        ~LineParser(void) {
            freeIfNotNULL(mKey);
        }


        inline const char *
        getKey(void) {
            return mKey;
        }


        inline ParseProcT
        getProc(void) {
            return mProc;
        }


        inline void *
        getContext(void) {
            return mCtx;
        }


    };





    class LineParserTable:
        public SynchronizedMap<std::string, LineParser *> {


    private:
        typedef std::map<std::string, LineParser *>::iterator CLPIterator;


        static void
        deleteHook(LineParser *v, void *arg) {
            (void)arg;
            if (v != NULL) {
                delete v;
            }
        }

        LineParserTable(const LineParserTable &obj);
        LineParserTable operator = (const LineParserTable &obj);


    public:
        LineParserTable() :
            SynchronizedMap<std::string, LineParser *>() {
            setDeleteHook(deleteHook, NULL);
        }


    };





private:


#define DefaultDelm	"\t\r\n "


    LineParserTable mTbl;
    char *mDelm;


    inline bool
    pParse(const char *path) {
        bool ret = false;

        int lineNo = 0;
        LineParser *p;
        std::string k = (const char *)"";
        ParseProcT proc;
        void *ctx;
        bool pFound;
        bool doLoop = true;
        char *orgBuf = NULL;
        char lineBuf[4096];
        char *tokens[4096];
        int nTokens;
        FILE *fd = NULL;

        if (isValidString(path) != true) {
            goto Done;
        }

        fd = fopen(path, "r");
        if (fd == NULL) {
            perror("fopen");
            goto Done;
        }

        while (fgets(lineBuf, sizeof(lineBuf), fd) != NULL &&
               doLoop == true) {
            lineNo++;
            proc = NULL;
            ctx = NULL;
            pFound = false;

            orgBuf = strdup(lineBuf);

            nTokens = getToken(lineBuf, tokens, 4096, mDelm);
            if (nTokens <= 0) {
                continue;
            }

            k = tokens[0];
            pFound = mTbl.get(k, p);
            if (pFound == true) {
                proc = p->getProc();
                ctx = p->getContext();

                doLoop = proc(path, lineNo, (const char *)orgBuf,
                              nTokens, (char * const *)tokens, ctx);
            } else {
                doLoop = onUnknown(path, lineNo, (const char *)orgBuf,
                                   nTokens, (char * const *)tokens, ctx);
            }
            freeIfNotNULL(orgBuf);
            orgBuf = NULL;

            if (doLoop == false) {
                goto Done;
            }
        }
        ret = true;

        Done:
        if (fd != NULL) {
            (void)fclose(fd);
        }
        freeIfNotNULL(orgBuf);

        return ret;
    }


    SimpleFileParser(const SimpleFileParser &obj);
    SimpleFileParser operator = (const SimpleFileParser &obj);


protected:
    inline bool
    addParser(const char *key, ParseProcT p, void *a) {
        LineParser *newProc =
            new LineParser(key, p, a);
        if (newProc == NULL) {
            fatal("Can't create a parser.\n");
        }
        std::string k = key;
        if (mTbl.put(k, newProc) != true) {
            fatal("Can't add a parser.\n");
        }
        return true;
    }


    inline void
    setDelimiter(const char *delm) {
        freeIfNotNULL(mDelm);
        mDelm = NULL;
        if (isValidString(delm) == false) {
            mDelm = strdup(DefaultDelm);
        } else {
            mDelm = strdup(delm);
        }
    }


    virtual inline bool
    onUnknown(const char *path, int lineNo,
              const char *line,
              int argc, char * const argv[], void *ctx) {
        (void)argc;
        (void)argv;
        (void)ctx;
        nata_MsgError("parse error:\n[%s:%d]\t%s\n",
                      path, lineNo, line);
        return false;
    }


public:


    SimpleFileParser(void) :
        // mTbl,
        mDelm(strdup(DefaultDelm)) {
    }


    virtual 
    ~SimpleFileParser(void) {
        freeIfNotNULL(mDelm);
    }


    static inline int
    getToken(char *buf, char **tokens, int max, const char *delm) {
        return nata_GetToken(buf, tokens, max, delm);
    }


    static inline bool
    parseInt32(const char *str, int32_t &val) {
        int32_t v;
        bool ret = nata_ParseInt32(str, &v);
        val = v;
        return ret;
    }

    
    static inline bool
    parseInt64(const char *str, int64_t &val) {
        int64_t v;
        bool ret = nata_ParseInt64(str, &v);
        val = v;
        return ret;
    }


    static inline bool
    parseFloat(const char *str, float &val) {
        float v;
        bool ret = nata_ParseFloat(str, &v);
        val = v;
        return ret;
    }


    static inline bool
    parseDouble(const char *str, double &val) {
        double v;
        bool ret = nata_ParseDouble(str, &v);
        val = v;
        return ret;
    }


    static inline bool
    parseLongDouble(const char *str, long double &val) {
        long double v;
        bool ret = nata_ParseLongDouble(str, &v);
        val = v;
        return ret;
    }


    static inline bool
    parseBool(const char *str, bool &val) {
        if (strcasecmp(str, "true") == 0 ||
            strcasecmp(str, "yes") == 0 ||
            strcasecmp(str, "on") == 0 ||
            strcasecmp(str, "enabled") == 0 ||
            strcasecmp(str, "1") == 0) {
            val = true;
            return true;
        } else if (strcasecmp(str, "false") == 0 ||
                   strcasecmp(str, "no") == 0 ||
                   strcasecmp(str, "off") == 0 ||
                   strcasecmp(str, "disabled") == 0 ||
                   strcasecmp(str, "0") == 0) {
            val = false;
            return true;
        } else {
            return false;
        }
    }


    inline bool
    parse(const char *path) {
        return pParse(path);
    }


#undef DefaultDelm
};


#endif // ! __SIMPLEFILEPARSER_H__
