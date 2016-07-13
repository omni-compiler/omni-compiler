package exc.openacc;

public enum AccDevice {
    /*                 maxNumGangs, maxNumWorkers, maxVectorLength, defaultVectorLength, RODataCacheAvailable*/
    Fermi("Fermi",           65535,             1,            1024,                 256,          false),
    Kepler("Kepler",    2147483647,             1,            1024,                 256,           true),
    Maxwell("Maxwell",  2147483647,             1,            1024,                 256,           true),
    NONE("",                     0,             0,               0,                   0,          false),
    ;
    //2^31-1 = 2147483647
    private String name;
    private int majorVersion;
    private int minorVersion;
    private int maxNumGangs;
    private int maxNumWorkers;
    private int maxVectorLength;
    private int defaultVectorLength;
    private boolean readOnlyDataCacheAvailable = true;
    private boolean useReadOnlyDataCache;
    private AccDevice(String name, int maxNumGangs, int maxNumWorkers, int maxVectorLength, int defaultVectorLength, boolean readOnlyDataCacheAvailable){
        this.name = name;
        this.maxNumGangs = maxNumGangs;
        this.maxNumWorkers = maxNumWorkers;
        this.maxVectorLength = maxVectorLength;
        this.defaultVectorLength = defaultVectorLength;
        this.readOnlyDataCacheAvailable = this.useReadOnlyDataCache = readOnlyDataCacheAvailable;
    }

    String getName()
    {
        return name;
    }
    int getMaxNumGangs()
    {
        return maxNumGangs;
    }
    int getMaxNumWorkers()
    {
        return maxNumWorkers;
    }
    int getMaxVectorLength()
    {
        return maxVectorLength;
    }

    public void setDefaultVectorLength(int defaultVectorLength)
    {
        if(defaultVectorLength <= 0 || defaultVectorLength > maxVectorLength){
            ACC.fatal("invalid vectorlength");
        }
        this.defaultVectorLength = defaultVectorLength;
    }
    int getDefaultVectorLength()
    {
        return defaultVectorLength;
    }

    public void setUseReadOnlyDataCache(boolean useReadOnlyDataCache)
    {

        if(!readOnlyDataCacheAvailable && useReadOnlyDataCache){
            ACC.warning("Read-only data cache is unavailable");
            return;
        }
        this.useReadOnlyDataCache = useReadOnlyDataCache;
    }
    boolean getUseReadOnlyDataCache()
    {
        return useReadOnlyDataCache;
    }
}
