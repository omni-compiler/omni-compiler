#include <nata/nata_rcsid.h>
__rcsId("$Id: SimpleFileParser.cpp 86 2012-07-30 05:33:07Z m-hirano $")

#include <nata/libnata.h>

#include <nata/SimpleFileParser.h>



// Class methods


char *
SimpleFileParser::strlowerdup(const char *str) {
    char *p = strdup(str);
    char *ret = p;

    while (isValidString(p) == true) {
	*p = (char)tolower((int)*p);
	p++;
    }

    return ret;
}


int
SimpleFileParser::strtokenize(char *buf, char **tokens, int max,
			      const char *delm) {
    return nata_GetToken(buf, tokens, max, delm);
}


char *
SimpleFileParser::strtrimrdup(const char *org, const char *trimChars) {
    return nata_TrimRight(org, trimChars);
}


bool
SimpleFileParser::parseInt32(const char *str, int32_t &val, int base) {
    return nata_ParseInt32ByBase(str, &val, base);
}


bool
SimpleFileParser::parseInt64(const char *str, int64_t &val, int base) {
    return nata_ParseInt64ByBase(str, &val, base);
}


bool
SimpleFileParser::parseBool(const char *str, bool &val) {
    if (strcasecmp(str, "true") == 0 ||
	strcasecmp(str, "yes") == 0 ||
	strcmp(str, "1") == 0) {
	val = true;
	return true;
    } else if (strcasecmp(str, "false") == 0 ||
	       strcasecmp(str, "no") == 0 ||
	       strcmp(str, "0") == 0) {
	val = false;
	return true;
    } else {
	return false;
    }
}


bool
SimpleFileParser::parseFloat(const char *str, float &val) {
    return nata_ParseFloat(str, &val);
}


bool
SimpleFileParser::parseDouble(const char *str, double &val) {
    return nata_ParseDouble(str, &val);
}


bool
SimpleFileParser::parseLongDouble(const char *str, long double &val) {
    return nata_ParseLongDouble(str, &val);
}



// Private methods


bool
SimpleFileParser::addParseProc(const char *ident,
			       SimpleFileParserProcT proc) {
    const char *lIdent = strlowerdup(ident);
    SimpleFileParserProcTableInsertResult r;
    std::string key = lIdent;

    r = mProcTblPtr->insert(SimpleFileParserProcTableInsertDatum(key, proc));

    (void)free((void *)lIdent);
    return r.second;
}


SimpleFileParserProcT
SimpleFileParser::findParseProc(const char *ident) {
    SimpleFileParserProcT ret = NULL;    
    const char *lIdent = strlowerdup(ident);

    SimpleFileParserProcTableIterator it = mProcTblPtr->find(lIdent);

    if (it != mProcTblPtr->end()) {
	ret = it->second;
    }

    (void)free((void *)lIdent);

    return ret;
}


bool
SimpleFileParser::parseLine(const char *lPtr) {
    char *line = strdup(lPtr);
    char **tokens = NULL;
    int nTokens = 0;
    SimpleFileParserProcT proc = NULL;
    bool procRet = false;
    bool beforeRet = false;
    bool afterRet = false;
    bool ret = false;

    if (isValidString(mDelimiter) == false) {
	return false;
    }

    tokens = (char **)alloca(sizeof(char *) * mNTokens);

    nTokens = tokenize(line, tokens, mNTokens, mDelimiter);
    if (nTokens == 0) {
	// An empty line.
	mLineState = LineState_OK;
	ret = true;
	goto Done;
    }
    if (mCommentLeader != '\0') {
        if (tokens[0][0] == mCommentLeader) {
            mLineState = LineState_OK;
            ret = true;
            goto Done;
        }
    }

    proc = findParseProc(tokens[0]);
    if (proc == NULL) {
	mLineState = LineState_NoProc;
	goto Done;
    }

    beforeRet = beforeParseHook();
    if (beforeRet == false) {
	mLineState = LineState_BeforeParseError;
	goto Done;
    }

    procRet = (proc)(this, nTokens, tokens);
    if (procRet == false) {
	mLineState = LineState_ParseError;
	goto Done;
    }

    afterRet = afterParseHook();
    if (beforeRet == false) {
	mLineState = LineState_AfterParseError;
	goto Done;
    }

    ret = true;

    Done:
    freeIfNotNULL(line);

    return ret;
}


void
SimpleFileParser::errorMessage(const char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    va_end(args);

    fprintf(stderr, "%s:line %d: ", mFileName, mCurLine);
    vfprintf(stderr, fmt, args);
}



// Public methods


void
SimpleFileParser::setDelimiter(const char *delimiter) {
    mDelimiter = delimiter;
}


bool
SimpleFileParser::parse(const char *fileName) {
    bool ret = false;

    if (isValidString(fileName) == false) {
	return false;
    }

    freeIfNotNULL(mFileName);
    mFileName = strdup(fileName);

    if (mFd != NULL) {
	(void)fclose(mFd);
	mFd = NULL;
    }
    mFd = fopen(fileName, "r");
    if (mFd == NULL) {
	perror("fopen");
	(void)free((void *)mFileName);
	mFileName = NULL;
	return false;
    }

    mCurLine = 0;
    char *lineBuf = (char *)alloca(mMaxLineLen);

    openHook();

    while (fgets(lineBuf, mMaxLineLen, mFd) != NULL) {
	mCurLine++;
	mLineState = LineState_Unknown;
	if (parseLine(lineBuf) == false) {
	    char *tLine = strtrimrdup(lineBuf, "\r\n");

	    switch (mLineState) {
		case LineState_NoProc: {
		    errorMessage("An unknown identifier found in '%s'.\n",
				 tLine);
		    break;
		}
		case LineState_ParseError: {
		    errorMessage("Parse error(s) in '%s'.\n",
				 tLine);
		    break;
		}
		case LineState_BeforeParseError:
		case LineState_AfterParseError: {
		    errorMessage("Context/Sequence mismatch(s) in '%s'.\n",
				 tLine);
		    break;
		}
		default: {
		    break;
		}
	    }

	    (void)free((void *)tLine);

	    goto Done;
	}
    }

    ret = true;

    Done:
    if (mFd != NULL) {
	(void)fclose(mFd);
	mFd = NULL;
    }
    ret = closeHook(ret);

    freeIfNotNULL(mFileName);
    mFileName = NULL;
    mCurLine = 0;

    return ret;
}


const char *
SimpleFileParser::getFileName(void) {
    return mFileName;
}


int
SimpleFileParser::getCurrentLineNumber(void) {
    return mCurLine;
}

