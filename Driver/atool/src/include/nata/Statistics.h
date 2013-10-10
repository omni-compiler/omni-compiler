/*
 * $Id: Statistics.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __STATISTICS_H__
#define __STATISTICS_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>



template<typename T>
class Statistics {



private:
    __rcsId("$Id: Statistics.h 86 2012-07-30 05:33:07Z m-hirano $");

    T mSum;
    T m2Sum;
    T mMin;
    T mMax;
    uint64_t mNRecs;
    Mutex mLock;



public:
    Statistics(void) :
        mSum(0),
        m2Sum(0),
        mMin((T)INT_MAX),
        mMax((T)INT_MIN),
        mNRecs(0) {
        (void)rcsid();
    }


    void
    record(T val) {
        mLock.lock();
        if (mNRecs > 0) {
            if (val < mMin) {
                mMin = val;
            }
            if (val > mMax) {
                mMax = val;
            }
        } else {
            mMax = mMin = val;
        }
        mSum += val;
        m2Sum += (val * val);
        mNRecs++;
        mLock.unlock();
    }

#ifdef min
#define oMin min
#undef min
#endif /* min */
    T
    min(void) {
        return mMin;
    }
#ifdef oMin
#define min oMin
#undef oMin
#endif /* oMin */


#ifdef max
#define oMax max
#undef max
#endif /* max */
    T
    max(void) {
        return mMax;
    }
#ifdef oMax
#define max oMax
#undef oMax
#endif /* oMax */


    double
    average(bool ignoreMinMax = false) {
        if (mNRecs > 0) {
            if (ignoreMinMax == false) {
                return (double)mSum / (double)mNRecs;
            } else {
                if (mNRecs < 3) {
                    return 0.0;
                } else {
                    return (double)(mSum - mMin - mMax) /
                        ((double)(mNRecs - 2));
                }
            }
        } else {
            return 0.0;
        }
    }


    double
    variance(bool ignoreMinMax = false) {
        if (mNRecs <= 1) {
            return 0.0;
        }
        double avg = average(ignoreMinMax);
        if (avg == 0.0) {
            return 0.0;
        } else {
            if (mNRecs < 1) {
                return 0.0;
            } else {
                if (ignoreMinMax == false) {
                    return ((double)m2Sum
                            - 2.0 * avg * (double)mSum
                            + avg * avg * (double)mNRecs) / 
                        ((double)(mNRecs - 1));
                } else {
                    if (mNRecs < 4) {
                        return 0.0;
                    } else {
                        return ((double)(m2Sum - mMin * mMin - mMax * mMax)
                                - 2.0 * avg * (double)(mSum - mMin - mMax)
                                + avg * avg * (double)(mNRecs - 2)) /
                            ((double)(mNRecs - 3));
                    }
                }
            }
        }
    }


    uint64_t
    recorded(void) {
        return mNRecs;
    }


    void
    reset(void) {
        mSum = (T)0;
        m2Sum = (T)0;
        mMin = (T)INT_MAX;
        mMax = (T)-INT_MAX;
        mNRecs = 0;
    }
};


#endif // ! __STATISTICS_H__
