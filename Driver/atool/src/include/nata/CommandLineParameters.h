/* 
 * $Id: CommandLineParameters.h 161 2012-08-22 08:35:48Z m-hirano $
 */
#ifndef __COMMANDLINEPARAMETERS_H__
#define __COMMANDLINEPARAMETERS_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>
#include <nata/nata_util.h>

#include <nata/nata_macros.h>

#include <nata/RegularExpression.h>
#include <nata/SynchronizedMap.h>
#include <nata/ScopedLock.h>

#include <string>
#include <vector>

#include <nata/nata_perror.h>





class CommandLineParameters {


private:
    __rcsId("$Id: CommandLineParameters.h 161 2012-08-22 08:35:48Z m-hirano $");


protected:


    typedef enum {
        Parse_Unknown = 0,
        Parse_This,
        Parse_Next,
        Parse_ThisWithValue
    } ParseType;


    typedef enum {
        Value_Unknown = 0,
        Value_None,
        Value_String,
        Value_Boolean,
        Value_Integer,
        Value_Float
    } ValueType;

        

private:


    typedef bool (*ParseProcT)(const char *arg, void *ctx);





    class CommandLineParser {
    private:
        const char *mKey;
        ParseProcT mProc;
        void *mCtx;
        ParseType mPType;
        ValueType mVType;
        bool mIsMandatory;
        bool mIsRegexp;
        const char *mUsage;


    public:
        CommandLineParser(const char *key,
                          ParseProcT p, void *ctx,
                          ParseType pt,
                          ValueType vt,
                          bool mandatory = false,
                          bool isRegexp = false,
                          const char *usage = NULL) :
            mKey(NULL),
            mProc(p),
            mCtx(ctx),
            mPType(pt),
            mVType(vt),
            mIsMandatory(mandatory),
            mIsRegexp(isRegexp),
            mUsage(usage) {

            if (isValidString(key) == true) {
                mKey = (const char *)strdup(key);
                if (mKey == NULL) {
                    fatal("Can't allocate a key.\n");
                }
            } else {
                fatal("The key is not a valid string.\n");
            }

            if (isValidString(usage) == true) {
                mUsage = (const char *)strdup(usage);
                if (mUsage == NULL) {
                    fatal("Can't allocate a usage.\n");
                }
            }

            if (mPType == Parse_Unknown) {
                fatal("Invalid parse type.\n");
            }
            if (mVType == Value_Unknown) {
                fatal("Invalid value type.\n");
            }
        }


        ~CommandLineParser(void) {
            freeIfNotNULL(mKey);
            freeIfNotNULL(mUsage);
        }


    private:
        CommandLineParser(const CommandLineParser &obj);
        CommandLineParser operator = (const CommandLineParser &obj);


    public:
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


        inline ParseType
        getParseType(void) {
            return mPType;
        }


        inline ValueType
        getValueType(void) {
            return mVType;
        }


        inline bool
        isMandatory(void) {
            return mIsMandatory;
        }


        inline const char *
        getUsage(void) {
            return mUsage;
        }
    };


    class CommandLineParserTable:
        public SynchronizedMap<std::string, CommandLineParser *> {


    private:
        typedef std::map<std::string, CommandLineParser *>::iterator
        CLPIterator;


        static int
        cmpProc(const void *v0, const void *v1) {
            CommandLineParameters::CommandLineParser **cp0 =
                (CommandLineParameters::CommandLineParser **)v0;
            CommandLineParameters::CommandLineParser **cp1 =
                (CommandLineParameters::CommandLineParser **)v1;

            return strcmp((*cp0)->getKey(), (*cp1)->getKey());
        }


        static void
        deleteHook(CommandLineParser *v, void *arg) {
            (void)arg;
            if (v != NULL) {
                delete v;
            }
        }

        CommandLineParserTable(const CommandLineParserTable &obj);
        CommandLineParserTable operator = (const CommandLineParserTable &obj);


    public:
        CommandLineParserTable() :
            SynchronizedMap<std::string, CommandLineParser *>() {
            setDeleteHook(deleteHook, NULL);
        }

        inline size_t
        getParsers(CommandLineParser **cpPtrs, const size_t ncpPtrs, 
                   bool doSort) {

            size_t nAdd = 0;
            CommandLineParser *cpPtr = NULL;

            lock();
            {
                CLPIterator it;
                CLPIterator endIt = end();

                for (it = begin(); it != endIt; it++) {
                    cpPtr = it->second;
                    if (cpPtr != NULL) {
                        if (nAdd < ncpPtrs) {
                            cpPtrs[nAdd] =  cpPtr;
                            nAdd++;
                        } else {
                            break;
                        }
                    }
                }
            }
            unlock();

            if (doSort == true) {
                qsort((void *)cpPtrs, nAdd, sizeof(cpPtr), cmpProc);
            }

            return nAdd;
        }
    };


    class RegularExpressionTable {
    private:
        typedef std::vector<RegularExpression *> regexpTable;

        regexpTable mTbl;

        RegularExpressionTable(const RegularExpressionTable &obj);
        RegularExpressionTable operator = (const RegularExpressionTable &obj);


    public:
        RegularExpressionTable(void) {
        }

        ~RegularExpressionTable(void) {
            size_t i;

            for (i = 0; i < mTbl.size(); i++) {
                delete mTbl[i];
            }
        }

        inline void
        add(const char *exp) {
            RegularExpression *rePtr = new RegularExpression(exp);
            if (rePtr != NULL) {
                mTbl.push_back(rePtr);
            }
        }

        inline const char *
        match(const char *exp) {
            size_t i;

            for (i = 0; i < mTbl.size(); i++) {
                if (mTbl[i]->match(exp) == true) {
                    return mTbl[i]->getExpression();
                }
            }

            return NULL;
        }

    };



private:
    CommandLineParserTable mTbl;
    RegularExpressionTable mRETbl;

    const char *mProgName;
    const char **mNewArgv;
    int mNewArgc;


private:
    void
    pSetProgName(const char *argv0) {
        const char *p = (const char *)strrchr(argv0, '/');
        if (p != NULL) {
            p++;
        } else {
            p = argv0;
        }
        mProgName = (const char *)strdup(p);
        if (mProgName == NULL) {
            fatal("Can't allocate my name.\n");
        }
    }


    inline bool
    pParse(int argc, char * const argv[]) {
        bool ret = false;
        mNewArgv = (const char **)malloc(sizeof(const char *) * argc);
        if (mNewArgv == NULL) {
            fatal("Can't allocate new argv.\n");
        }
        mNewArgc = 0;

        CommandLineParser *p;
        std::string k = (const char *)"";

        ParseProcT proc;
        void *ctx;
        ParseType pt;
        ValueType vt;
        char *tmpP;
        char *orgTmpP;
        bool pFound;
        bool hasEqual;
        bool regMatched;
        const char *valPtr = NULL;
        const char *exp = NULL;
        bool doLoop = true;

        while (*argv != NULL && doLoop == true) {
            p = NULL;
            proc = NULL;
            ctx = NULL;
            pt = Parse_Unknown;
            vt = Value_Unknown;
            pFound = false;
            hasEqual = false;
            regMatched = false;
            freeIfNotNULL(valPtr);
            valPtr = NULL;

            k = *argv;
            if ((pFound = mTbl.get(k, p)) == true) {
                //
                // full-exact-matched keyword.
                //
                ExactMatch:
                pt = p->getParseType();
                switch (pt) {
                    case Parse_Next: {
                        argv++;
                        if (*argv != NULL) {
                            valPtr = (const char *)strdup(*argv);
                        } else {
                            valPtr = NULL;
                        }
                        break;
                    }
                    case Parse_This: {
                        valPtr = (const char *)strdup(*argv);
                        break;
                    }
                    case Parse_ThisWithValue: {
                        //
                        // expected "--XXX=VVV", but got "--XXX="
                        //
                        goto NotFound;
                    }
                    default: {
                        fatal("Something wrong.\n");
                        break;
                    }
                }
            } else if ((orgTmpP = strchr(*argv, '=')) != NULL) {
                //
                // Then check if it is like "--XXX=VVV"
                //
                EqualMatch:
                char *cpK = strdup(*argv);
                tmpP = strchr(cpK, '=');
                tmpP++;
                *tmpP = '\0';
                k = cpK;
                (void)free((void *)cpK);
                hasEqual = true;

                if ((pFound = mTbl.get(k, p)) == true) {
                    pt = p->getParseType();
                    if (pt == Parse_ThisWithValue) {
                        valPtr = (const char *)strdup(++orgTmpP);
                    } else {
                        goto NotFound;
                    }
                } else {
                    //
                    // The last chance remains in regexp match.
                    //
                    if ((exp = mRETbl.match(*argv)) != NULL &&
                        regMatched == false) {
                        goto RegMatch;
                    } else {
                        goto NotFound;
                    }
                }
            } else if ((exp = mRETbl.match(*argv)) != NULL) {
                //
                // regex-matched keyword.
                //
                RegMatch:
                k = exp;
                regMatched = true;

                if ((pFound = mTbl.get(k, p)) == true) {
                    goto ExactMatch;
                } else if ((orgTmpP = strchr(*argv, '=')) != NULL &&
                           hasEqual == false) {
                    goto EqualMatch;
                } else {
                    goto NotFound;
                }
            } else {
                NotFound:
                pFound = false;
                valPtr = (const char *)strdup(*argv);
            }

            if (pFound == true) {
                if (valPtr != NULL) {
                    bool doProc = false;
                    proc = p->getProc();
                    ctx = p->getContext();
                    vt = p->getValueType();

                    if (pt != Parse_This) {
                        switch (vt) {
                            case Value_String: {
                                doProc = true;
                                break;
                            }

                            case Value_Boolean: {
                                bool val;
                                if (parseBool(valPtr, val) == true) {
                                    doProc = true;
                                } else {
                                    fprintf(stderr,
                                            "%s: error: the '%s' option "
                                            "requires a boolean value "
                                            "but '%s' doesn't seem as.\n",
                                            mProgName, k.c_str(), valPtr);
                                    doLoop = false;
                                }
                                break;
                            }

                            case Value_Integer: {
                                int64_t val;
                                if (parseInt64(valPtr, val) == true) {
                                    doProc = true;
                                } else {
                                    fprintf(stderr,
                                            "%s: error: the '%s' option "
                                            "requires an integer value "
                                            "but '%s' doesn't seem as.\n",
                                            mProgName, k.c_str(), valPtr);
                                    doLoop = false;
                                }
                                break;
                            }

                            case Value_Float: {
                                long double val;
                                if (parseLongDouble(valPtr, val) == true) {
                                    doProc = true;
                                } else {
                                    fprintf(stderr,
                                            "%s: error: the '%s' option "
                                            "requires a floating point value "
                                            "but '%s' doesn't seem as.\n",
                                            mProgName, k.c_str(), valPtr);
                                    doLoop = false;
                                }
                                break;
                            }

                            default: {
                                break;
                            }
                        }
                    } else {
                        doProc = true;
                    }

                    if (doProc == true && proc != NULL) {
                        doLoop = (proc)(valPtr, ctx);
                    }

                } else {
                    fprintf(stderr,
                            "%s: error: need a value for "
                            "'%s' option.\n",
                            mProgName,
                            k.c_str());
                    doLoop = false;
                }

            } else {
                mNewArgv[mNewArgc++] = (const char *)strdup(valPtr);
            }

            if (doLoop == false) {
                goto Done;
            }

            argv++;
        }
        ret = true;

        Done:
        freeIfNotNULL(valPtr);

        return ret;
    }


protected:
    inline bool
    addParser(const char *key, ParseProcT p, void *a,
              ParseType pt,
              ValueType vt,
              bool mandatory = false,
              bool isRegexp = false,
              const char *comm = NULL) {
        CommandLineParser *newProc =
            new CommandLineParser(key, p, a, pt, vt, mandatory, isRegexp, 
                                  comm);
        if (newProc == NULL) {
            fatal("Can't create a parser.\n");
        }
        std::string k = key;
        if (mTbl.put(k, newProc) != true) {
            fatal("Can't add a parser.\n");
        }
        if (isRegexp == true) {
            mRETbl.add(key);
        }

        return true;
    }


public:


    CommandLineParameters(const char *argv0) :
        mProgName(NULL),
        mNewArgv(NULL),
        mNewArgc(0) {
        pSetProgName(argv0);
    }


    virtual 
    ~CommandLineParameters(void) {
        freeIfNotNULL(mProgName);
        if (mNewArgv != NULL && mNewArgc > 0) {
            for (int i = 0; i < mNewArgc; i++) {
                freeIfNotNULL(mNewArgv[i]);
            }
        }
    }


private:
    CommandLineParameters(const CommandLineParameters &obj);
    CommandLineParameters operator = (const CommandLineParameters &obj);


public:


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


public:
    inline const char *
    commandName(void) {
        return mProgName;
    }


    inline bool
    parse(int argc, char * const argv[]) {
        return pParse(argc, argv);
    }


    inline virtual void
    usage(bool doHeader) {
        if (doHeader == true) {
            fprintf(stderr, "Usage:\n\t%s OPTIONS...\n", mProgName);
        }

        ParseType pt;
        ValueType vt;
        bool isMand;
        const char *usageStr;
        const char *key;

        CommandLineParser *parsers[1024];
        size_t nParsers = mTbl.getParsers(parsers, 1024, true);

        std::string tmp = "";

        if (nParsers > 0) {
            fprintf(stderr, "Where OPTIONS are:\n");

            for (size_t i = 0; i < nParsers; i++) {
                pt = parsers[i]->getParseType();
                vt = parsers[i]->getValueType();
                isMand = parsers[i]->isMandatory();
                usageStr = parsers[i]->getUsage();
                key = parsers[i]->getKey();

                tmp = key;

                switch (pt) {
                    case Parse_This: {
                        break;
                    }
                    case Parse_Next:
                    case Parse_ThisWithValue: {
                        if (pt == Parse_Next) {
                            tmp.append(" ");
                        }
                        switch (vt) {
                            case Value_String: {
                                tmp.append("str");
                                break;
                            }
                            case Value_Boolean: {
                                tmp.append("bool");
                                break;
                            }
                            case Value_Integer: {
                                tmp.append("#");
                                break;
                            }
                            case Value_Float: {
                                tmp.append("#.#");
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        break;
                    }
                    default: {
                        break;
                    }
                }

                if (isMand == false) {
                    tmp.insert(0, "[");
                    tmp.append("]");
                } else {
                    tmp.insert(0, " ");
                    tmp.append(" ");
                }

                tmp.insert(0, "  ");
                tmp.append("\t");
                tmp.append(usageStr);

                fprintf(stderr, "%s\n", tmp.c_str());
            }
        }
    }


    inline void
    getUnparsedOptions(int &argc, const char **&argv) {
        argc = mNewArgc;
        argv = mNewArgv;
    }
        

};


#endif // ! __COMMANDLINEPARAMETERS_H__

