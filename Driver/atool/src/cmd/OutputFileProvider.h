/*
 * $Id: OutputFileProvider.h 235 2013-09-06 03:42:51Z m-hirano $
 */
#ifndef __OUTPUTFILEPROVIDER_H__
#define __OUTPUTFILEPROVIDER_H__

#include <nata/nata_rcsid.h>

#include <nata/libnata.h>

#include <nata/Mutex.h>
#include <nata/ScopedLock.h>
#include <nata/SynchronizedMap.h>

#include <vector>

#include <nata/nata_perror.h>


namespace OutputFileProviderStatics {
    extern void		sSafeUnlink(const char *path);
    extern int		sOpenOutputFile(const char *path);
    extern bool		sRewindFile(int fd);
    extern const char *	sGenerateTemporaryName(const char *prefix,
                                               const char *suffix);
}


class OutputFileProvider {


private:


    __rcsId("$Id: OutputFileProvider.h 235 2013-09-06 03:42:51Z m-hirano $")





    class mOutputFile {


    private:


        Mutex mLock;
        const char *mPath;
        int mFd;
        FILE *mStdFd;
        bool mIsTemp;

        mOutputFile(const mOutputFile &obj);
        mOutputFile operator = (const mOutputFile &obj);


        inline void
        mCleanup(void) {
            if (isValidString(mPath) == true) {
                if (mIsTemp == true) {
                    OutputFileProviderStatics::sSafeUnlink(mPath);
                }
            }
            if (mStdFd != NULL) {
                (void)::fclose(mStdFd);
            } else if (mFd >= 0) {
                (void)::close(mFd);
            }

            free((void *)mPath);
            mPath = NULL;
            mFd = -INT_MAX;
        }


        inline void
        mCreate(const char *path, bool isTemp) {
            mCleanup();
            mPath = (const char *)strdup(path);
            mIsTemp = isTemp;
        }


        inline void
        mCreateTemp(const char *prefix, const char *suffix,
                    bool isTemp = true) {
            mCleanup();
            mPath =
                OutputFileProviderStatics::sGenerateTemporaryName(prefix,
                                                                  suffix);
            mIsTemp = isTemp;
        }


        inline int
        mOpen(void) {
            if (mFd < 0) {
                mFd = OutputFileProviderStatics::sOpenOutputFile(mPath);
            }
            return mFd;
        }


        inline bool
        mRewind(void) {
            return OutputFileProviderStatics::sRewindFile(mFd);
        }


        inline void
        mClose(void) {
            mFclose();
        }


        inline void
        mUnlink(void) {
            OutputFileProviderStatics::sSafeUnlink(mPath);
            (void)free((void *)mPath);
            mPath = NULL;
        }


        inline void
        mDestroy(void) {
            mClose();
            mUnlink();
        }


        inline FILE *
        mFopen(const char *mode) {
            if (mStdFd == NULL) {
                if (mFd < 0) {
                    if (mOpen() >= 0) {
                        if ((mStdFd = fdopen(mFd, mode)) != NULL) {
                            (void)setvbuf(mStdFd, NULL, _IONBF, 0);
                        }
                    }
                }
            }
            return mStdFd;
        }


        inline void
        mFclose(void) {
            if (mStdFd != NULL) {
                (void)::fclose(mStdFd);
            } else {
                if (mFd >= 0) {
                    (void)::close(mFd);
                }
            }
            mStdFd = NULL;
            mFd = -INT_MAX;
            if (mIsTemp == true) {
                mUnlink();
            }
        }

        
    public:


        mOutputFile(const char *path, bool isTemp = false) :
            // mLock,
            mPath(NULL),
            mFd(-INT_MAX),
            mStdFd(NULL),
            mIsTemp(false) {
            mCreate(path, isTemp);
        }

        mOutputFile(const char *prefix, const char *suffix,
                    bool isTemp = true) :
            // mLock,
            mPath(NULL),
            mFd(-INT_MAX),
            mStdFd(NULL),
            mIsTemp(isTemp) {
            mCreateTemp(prefix, suffix, isTemp);
        }

        ~mOutputFile(void) {
            ScopedLock l(&mLock);
            mCleanup();
        }


        inline int
        open(void) {
            ScopedLock l(&mLock);
            return mOpen();
        }


        inline FILE *
        fopen(const char *mode) {
            ScopedLock l(&mLock);
            return mFopen(mode);
        }


        inline bool
        rewind(void) {
            ScopedLock l(&mLock);
            return mRewind();
        }


        inline void
        close(void) {
            ScopedLock l(&mLock);
            return mClose();
        }


        inline void
        fclose(void) {
            ScopedLock l(&mLock);
            return mClose();
        }


        inline void
        unlink(void) {
            ScopedLock l(&mLock);
            return mUnlink();
        }


        inline void
        destroy(void) {
            ScopedLock l(&mLock);
            return mDestroy();
        }


        inline int
        getFd(void) {
            ScopedLock l(&mLock);
            return mFd;
        }


        inline const char *
        getPath(void) {
            ScopedLock l(&mLock);
            return mPath;
        }


        inline bool
        isTemporary(void) {
            ScopedLock l(&mLock);
            return mIsTemp;
        }


    };





    class mOutputFileTable: public SynchronizedMap<int, mOutputFile *> {


    private:


        typedef std::map<int, mOutputFile *>::iterator OFIterator;


        inline void
        mDestroy(void) {
            lock();
            {
                OFIterator it;
                OFIterator endIt = end();
                mOutputFile *ofPtr;

                for (it = begin(); it != endIt; it++) {
                    ofPtr = it->second;
                    ofPtr->destroy();
                }
            }
            unlock();
        }


        static void
        deleteHook(mOutputFile *ofPtr, void *arg) {
            (void)arg;
            if (ofPtr != NULL) {
                delete ofPtr;
            }
        }


        mOutputFileTable(const mOutputFileTable &obj);
        mOutputFileTable operator = (const mOutputFileTable &obj);


    public:


        mOutputFileTable(void) :
            SynchronizedMap<int, mOutputFile *>() {
            setDeleteHook(deleteHook, NULL);
            setRemoveHook(deleteHook, NULL);
        }


        inline void
        destroy(void) {
            mDestroy();
        }

        
    };





    Mutex mLock;
    mOutputFileTable mOFTbl;

    typedef std::vector<const char *> mFiles;
    mFiles mFilesBefore;
    bool mNeedCleanup;
    const char *mCwd;

    OutputFileProvider(const OutputFileProvider &obj);
    OutputFileProvider operator = (const OutputFileProvider &obj);





    inline void
    mCheckFiles(mFiles &list, const char *cwd) {
        struct dirent dent;
        struct dirent *resPtr;
        DIR *dir = opendir(cwd);
        if (dir != NULL) {
            while (readdir_r(dir, &dent, &resPtr) == 0 &&
                   resPtr != NULL) {
                list.push_back(strdup(dent.d_name));
            }
            ::closedir(dir);
        }
    }


    inline int
    mOpenAndAdd(mOutputFile *ofPtr) {
        int ret = -INT_MAX;
        if (ofPtr != NULL) {
            ret = ofPtr->open();
            if (ret >= 0) {
                mOFTbl.put(ret, ofPtr);
            } else {
                delete ofPtr;
            }
        } 
        return ret;
    }


    inline int
    mOpen(const char *path, bool isTemp) {
        bool isReallyTemp = 
            (mNeedCleanup == false) ? false : isTemp;
        mOutputFile *ofPtr = new mOutputFile(path, isReallyTemp);
        return mOpenAndAdd(ofPtr);
    }


    inline int
    mOpenTemp(const char *prefix, const char *suffix) {
        mOutputFile *ofPtr = new mOutputFile(prefix, suffix, mNeedCleanup);
        return mOpenAndAdd(ofPtr);
    }


    inline bool
    mRewind(int fd) {
        bool ret = false;
        mOutputFile *ofPtr = NULL;
        if (mOFTbl.get(fd, ofPtr) == true) {
            ret = ofPtr->rewind();
        }
        return ret;
    }


    inline void
    mClose(int fd) {
        mOutputFile *ofPtr = NULL;
       if (mOFTbl.get(fd, ofPtr) == true) {
            ofPtr->close();
            mOFTbl.remove(fd, ofPtr);
        }
    }


    inline FILE *
    mFopen(int fd, const char *mode) {
        FILE *ret = NULL;
        mOutputFile *ofPtr = NULL;
        if (mOFTbl.get(fd, ofPtr) == true) {
            ret = ofPtr->fopen(mode);
        }
        return ret;
    }


    inline void
    mFclose(int fd) {
        mOutputFile *ofPtr = NULL;
        if (mOFTbl.get(fd, ofPtr) == true) {
            ofPtr->fclose();
            mOFTbl.remove(fd, ofPtr);
        }
    }


    inline void
    mDestroy(int fd) {
        mOutputFile *ofPtr = NULL;
        if (mOFTbl.get(fd, ofPtr) == true) {
            ofPtr->destroy();
            mOFTbl.remove(fd, ofPtr);
        }
    }


    inline void
    mUnlink(int fd) {
        mOutputFile *ofPtr = NULL;
        if (mOFTbl.get(fd, ofPtr) == true) {
            ofPtr->unlink();
        }
    }


    inline const char *
    mGetPath(int fd) {
        const char *ret = NULL;
        mOutputFile *ofPtr = NULL;
        if (mOFTbl.get(fd, ofPtr) == true) {
            ret = ofPtr->getPath();
        }
        return ret;
    }
        

    inline bool
    mIsTemporary(int fd) {
        bool ret = false;
        mOutputFile *ofPtr = NULL;
        if (mOFTbl.get(fd, ofPtr) == true) {
            ret = ofPtr->isTemporary();
        }
        return ret;
    }
        

    inline void
    mAbort(void) {
        mOFTbl.destroy();
        mFiles filesNow;

        if (mNeedCleanup == true) {
            mCheckFiles(filesNow, mCwd);

            size_t i, j;
            bool existed;

            for (i = 0; i < filesNow.size(); i++) {
                if (strcmp("core", filesNow[i]) == 0) {
                    continue;
                }
                existed = false;
                for (j = 0; j < mFilesBefore.size(); j++) {
                    if (strcmp(filesNow[i], mFilesBefore[j]) == 0) {
                        existed = true;
                        break;
                    }
                }
                if (existed == false) {
#if 0
                    fprintf(stderr, "'%s' must be deleted.\n", filesNow[i]);
#endif
                    safeUnlink(filesNow[i]);
                }
            }

            for (i = 0; i < filesNow.size(); i++) {
                free((void *)filesNow[i]);
            }
        }
    }





public:


    OutputFileProvider(bool needCleanup = true) : 
        // mLock,
        // mOFTbl,
        // mFilesBefore,
        mNeedCleanup(needCleanup),
        mCwd(NULL) {
        char buf[PATH_MAX];
        (void)getcwd(buf, sizeof(buf));
        mCwd = strdup(buf);
        mCheckFiles(mFilesBefore, mCwd);
    }


    ~OutputFileProvider(void) {
        size_t i;

        for (i = 0; i < mFilesBefore.size(); i++) {
            free((void *)mFilesBefore[i]);
        }

        free((void *)mCwd);
    }


    inline int
    open(const char *path, bool isTemp = false) {
        ScopedLock l(&mLock);
        return mOpen(path, isTemp);
    }


    inline int
    openTemp(const char *prefix, const char *suffix) {
        ScopedLock l(&mLock);
        return mOpenTemp(prefix, suffix);
    }


    inline bool
    rewind(int fd) {
        ScopedLock l(&mLock);
        return mRewind(fd);
    }


    inline FILE *
    fopen(int fd, const char *mode) {
        ScopedLock l(&mLock);
        return mFopen(fd, mode);
    }
        

    inline void
    close(int fd) {
        ScopedLock l(&mLock);
        mClose(fd);
    }


    inline void
    destroy(int fd) {
        ScopedLock l(&mLock);
        mDestroy(fd);
    }


    inline void
    unlink(int fd) {
        ScopedLock l(&mLock);
        mUnlink(fd);
    }


    inline const char *
    getPath(int fd) {
        ScopedLock l(&mLock);
        return mGetPath(fd);
    }


    inline bool
    isTemporary(int fd) {
        ScopedLock l(&mLock);
        return mIsTemporary(fd);
    }


    inline void
    abort(void) {
        ScopedLock l(&mLock);
        mAbort();
    }


    inline void
    setCleanupNeeded(bool val) {
        ScopedLock l(&mLock);
        mNeedCleanup = val;
    }


    static inline const char *
    getFileBasename(const char *file,
                    const char *suffix = NULL,
                    bool removeDir = true) {
        char *ret = NULL;
        const char *startPtr = file;
        if (removeDir == true) {
            const char *lastSlash = strrchr(file, '/');
            if (lastSlash != NULL) {
                startPtr = ++lastSlash;
            }
        }
        if (isValidString(startPtr) == true) {
            ret = strdup(startPtr);
            if (isValidString(suffix) == false) {
                char *lastDot = (char *)strrchr(ret, '.');
                if (lastDot != NULL) {
                    *lastDot = '\0';
                }
            } else {
                size_t suffixLen = strlen(suffix);
                size_t wholeLen = strlen(ret);

                if (wholeLen > suffixLen) {
                    size_t offsetMax = wholeLen - suffixLen;
                    if (strcmp(&(ret[offsetMax]), suffix) == 0) {
                        ret[offsetMax] = '\0';
                    }
                }
            }
        }
        return (const char *)ret;
    }


    static inline void
    safeUnlink(const char *path) {
        OutputFileProviderStatics::sSafeUnlink(path);
    }


    static inline const char *
    tempnam(const char *prefix, const char *suffix) {
        return OutputFileProviderStatics::sGenerateTemporaryName(prefix,
                                                                 suffix);
    }


};

#endif // __OUTPUTFILEPROVIDER_H__
