#include <nata/nata_rcsid.h>

#include <nata/libnata.h>

#include <nata/nata_perror.h>


namespace OutputFileProviderStatics {


    __rcsId("$Id: OutputFileProviderStatics.cpp 129 2012-08-10 13:22:07Z m-hirano $")





    static inline bool
    sIsFileUsableAsOutput(const char *path) {
        bool ret = false;
        struct stat s;
        int st = stat(path, &s);

        if (st < 0) {
            if (errno == ENOENT) {
                ret = true;
            } else {
                perror("stat");
            }
        } else if (S_ISREG(s.st_mode)) {
            ret = true;
        }

        return ret;
    }


    inline void
    sSafeUnlink(const char *path) {
        if (isValidString(path) == true) {
            struct stat s;
            int st = stat(path, &s);
            if (st == 0) {
                if (!S_ISDIR(s.st_mode)) {
                    (void)::unlink(path);
                }
            }
        }
    }


    inline int
    sOpenOutputFile(const char *path) {
        int ret = -INT_MAX;

        if (sIsFileUsableAsOutput(path) == true) {
            //
            // Use O_RDWR since we need to access the file like write,
            // rewined and read sequence.
            //
            ret = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (ret < 0) {
                perror("open");
            }
        }

        return ret;
    }


    inline bool
    sRewindFile(int fd) {
        off_t offset = ::lseek(fd, (off_t)0, SEEK_SET);
        if (offset < (off_t)0) {
            perror("lseek");
        }
        return (offset == (off_t)0) ? true : false;
    }
                                   

    inline const char *
    sGenerateTemporaryName(const char *prefix, const char *suffix) {
        char buf[PATH_MAX];
        nata_Uid uid;
        nata_getUid(&uid);

        snprintf(buf, sizeof(buf), "%s%s%s",
                 (prefix != NULL) ? prefix : "",
                 (char *)uid,
                 (suffix != NULL) ? suffix : "");

        return (const char *)strdup(buf);
    }


    class sUidInit {
    public:
        sUidInit(void) {
            nata_initUid();
        }
    };


    static sUidInit sUI;

}
