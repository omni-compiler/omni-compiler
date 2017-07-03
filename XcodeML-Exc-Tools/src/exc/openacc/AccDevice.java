package exc.openacc;

public enum AccDevice {
    /*                 maxNumGangs, maxNumWorkers, maxVectorLength, defaultVectorLength, RODataCacheAvailable*/
    NVIDIA("NVIDIA",             0,             0,               0,                   0,          false),
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

    public static AccDevice getDevice(String deviceName){
        if(deviceName.matches("^[cC][cC]\\d\\d$")){
            return getNvidiaDevice(Integer.parseInt(deviceName.substring(2)));
        }else{
            switch(deviceName){
            case "Fermi":
                return getNvidiaDevice(20);
            case "Kepler":
                return getNvidiaDevice(35);
            case "Maxwell":
                return getNvidiaDevice(52);
            case "Pascal":
                return getNvidiaDevice(60);
            default:
                return AccDevice.valueOf(deviceName);
            }
        }
    }

    static AccDevice getNvidiaDevice(int ccVersion){
        AccDevice d = AccDevice.NVIDIA;

        int majorVersion = ccVersion / 10;
        int minorVersion = ccVersion % 10;

        d.majorVersion = majorVersion;
        d.minorVersion = minorVersion;
        d.maxNumGangs = (majorVersion <= 2)? 65535 : 2147483647;
        d.maxNumWorkers = 1;
        d.maxVectorLength = 1024;
        d.defaultVectorLength = 256;
        d.useReadOnlyDataCache = d.readOnlyDataCacheAvailable = (ccVersion >= 35);

        return d;
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
