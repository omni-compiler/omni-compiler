#include <nata/nata_rcsid.h>
__rcsId("$Id: main.cpp 234 2013-09-03 09:06:40Z m-hirano $")

#include <nata/libnata.h>

#include <nata/Process.h>
#include <nata/ProcessJanitor.h>
#include <nata/SignalThread.h>

#include "CompileOptions.h"
#include "PipelineBuilder.h"
#include "OutputFileProvider.h"

#include <nata/nata_perror.h>


static OutputFileProvider ofp;
static bool sDoCleanup = false;

static void
finalize(int sig) {
    (void)sig;
    if (sDoCleanup == true) {
        ofp.abort();
    }
    exit(1);
}


static void
setupSignals() {
    SignalThread *st = new SignalThread();

    st->ignore(SIGPIPE);
    st->setHandler(SIGINT, finalize);
    st->setHandler(SIGTERM, finalize);
    st->start(false, false);
}


int
main(int argc, char *argv[]) {
    int ret = 1;

    nata_InitializeLogger(emit_Unknown, "", true, false, 0);
    setupSignals();
    ProcessJanitor::initialize();

    CompileOptions opts(argv[0]);
    if (opts.parse(argc - 1, argv + 1) != true) {
        return (opts.isUsageEmitted() == true) ? 0 : 1;
    }
    if (opts.validateOptions() != true) {
        return 1;
    }
    CompileOptions::CompileStageT stage = opts.getStage();
    const char *outputFile = opts.getOutputFile();
    bool isVerbose = opts.isVerbose();
    bool isDryrun = opts.isDryrun();

    sDoCleanup = opts.needCleanup();
    ofp.setCleanupNeeded(sDoCleanup);

    PipelineBuilder pb(&opts,
                       &ofp,
                       opts.getCppCommand(),
                       NULL,
                       opts.getTranslatorCommand(),
                       NULL,
                       opts.getNativeCompilerCommand());

    size_t i;
    char **srcs;
    size_t nSrcs = opts.getSourceFiles(srcs);

    PipelinedCommands *pPtr;
    const char *sfx = pb.getOutputFileSuffix(false);
    int xmpFd;
    const char *xmpFile;
    int cppFd;

    if (opts.onlyPrintSources() == true) {
        char absPath[PATH_MAX];
        for (i = 0; i < nSrcs; i++) {
            if (realpath(srcs[i], absPath) == absPath) {
                fprintf(stdout, "%s\n", absPath);
            } else {
                fprintf(stdout, "%s\n", srcs[i]);
            }
        }
        (void)fflush(stdout);
        return 0;
    }

    for (i = 0; i < nSrcs; i++) {

        cppFd = -INT_MAX;
        xmpFd = -INT_MAX;

        pPtr = pb.getXmpPipeline(srcs[i], cppFd);
        if (pPtr != NULL) {
            if ((int)stage <= CompileOptions::Stage_Decompile) {
                if (isValidString(outputFile) == true) {
                    //
                    // use the specified file as th final target.
                    //
                    xmpFd = ofp.open(outputFile, false);
                } else {
                    //
                    // use default final output filename.
                    //
                    const char *fName = 
                        CompileOptions::getDefaultOutputFilename(
                            srcs[i], sfx);
                    xmpFd = ofp.open(fName, false);
                    free((void *)fName);
                }
            } else {
                //
                // use default temporary file.
                //
                xmpFd = ofp.openTemp("/tmp/.", sfx);
            }

            pPtr->setOutFd(xmpFd);

            if (isVerbose == true) {
                pPtr->printCommands(stderr);
            }
            if (isDryrun == false) {
                if (pPtr->start(true, true) == false) {
                    pb.cleanupForBogusCpp(srcs[i]);
                    goto Done;
                }
            }
            pb.cleanupForBogusCpp(srcs[i]);
            delete pPtr;
            pPtr = NULL;
        }

        if (xmpFd > 0 && xmpFd != 1) {
            if ((xmpFile = ofp.getPath(xmpFd)) != NULL &&
                (pPtr = pb.getNativePipeline(srcs[i], xmpFile)) != NULL) {
                if (isVerbose == true) {
                    pPtr->printCommands(stderr);
                }
                if (isDryrun == false) {
                    if (pPtr->start(true, true) == false) {
                        goto Done;
                    }
                }
                delete pPtr;
                pPtr = NULL;
            }

            if (isDryrun == false) {
                ofp.close(xmpFd);
            } else {
                ofp.destroy(xmpFd);
            }
        }

        if (cppFd > 1) {
            if (isDryrun == false) {
                ofp.close(cppFd);
            } else {
                ofp.destroy(cppFd);
            }
        }
    }

    pPtr = pb.getPrelinkPipeline();
    if (pPtr != NULL) {
        if (isVerbose == true) {
            pPtr->printCommands(stderr);
        }
        if (isDryrun == false) {
            if (pPtr->start(true, true) == false) {
                goto Done;
            }
        }
        delete pPtr;
        pPtr = NULL;
    }

    pPtr = pb.getNativeLinkPipeline();
    if (pPtr != NULL) {
        if (isVerbose == true) {
            pPtr->printCommands(stderr);
        }
        if (isDryrun == false) {
            if (pPtr->start(true, true) == false) {
                goto Done;
            }
        }
        delete pPtr;
        pPtr = NULL;
    }
    ret = 0;

    Done:
    if (pPtr != NULL) {
        if (ret != 0) {
            int finalECode = -INT_MAX;
            size_t n = pPtr->getCommandsNumber();
            const char * const *pArgv = NULL;
            Process::ProcessFinalStateT st;

            for (i = 0; i < n; i++) {
                pArgv = pPtr->getCommandArguments(i);
                st = pPtr->getCommandFinalState(i);
                switch (st) {
                    case Process::Process_Finally_Exited: {
                        int eCode = pPtr->getCommandExitCode(i);
                        if (eCode != 0) {
                            fprintf(stderr, "%s: error: %s exited "
                                    "with exit code %d.\n",
                                    opts.commandName(),
                                    pArgv[0],
                                    eCode);
                            if (finalECode == -INT_MAX) {
                                finalECode = eCode;
                            }
                        }
                        break;
                    }
                    case Process::Process_Finally_Signaled: {
                        int sig = pPtr->getCommandExitSignal(i);
                        fprintf(stderr, "%s: error: %s got signal %d, %s%s\n",
                                opts.commandName(),
                                pArgv[0],
                                sig,
                                strsignal(sig),
                                (pPtr->commandDumpedCore(i) == true) ?
                                " (core dumped)." : ".");
                        break;
                    }
                    default: {
                        fatal("not expected to enter here.\n");
                        break;
                    }
                }
            }

            if (finalECode != -INT_MAX) {
                ret = finalECode;
            }

        }

        delete pPtr;
    }

    if (sDoCleanup == true) {
        if (ret != 0) {
            ofp.abort();
        }
        pb.unlinkPrelinkerOutput();
    }

    return ret;
}
