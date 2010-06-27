/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmLanguage;

/**
 * Decompile option.
 */
public class XmOption
{
    public static final int COMP_VENDOR_GNU = 'G';
    public static final int COMP_VENDOR_INTEL = 'I';

    /** if suppress to write line directives */
    private static boolean _suppressLineDirective = false;

    /** if compiling Xcalable MP is enabled */
    private static boolean _xcalableMP = false;

    /** if compiling Xcalable MP is enabled */
    private static boolean _openMP = false;

    /** if debug output is enabled */
    private static boolean _debugOutput = false;

    /** target language ID */
    private static XmLanguage _language = XmLanguage.C;
    
    /** if transforming Fortran IO statement as atomic operation */
    private static boolean _isAtomicIO = false;

    /** backend compiler vendor */
    private static int _compilerVendor = COMP_VENDOR_GNU;
    
    private XmOption()
    {
    }

    /**
     * Sets compiler to or not to suppress to write line directives.
     *
     * @param enable true then compiler suppress to write line directives.
     */
    public static void setIsSuppressLineDirective(boolean enable)
    {
        _suppressLineDirective = enable;
    }

    /**
     * Checks does decompiler suppress line directives.
     *
     * @return true if compiler suppress to write line directives.
     */
    public static boolean isSuppressLineDirective()
    {
        return _suppressLineDirective;
    }

    /**
     * Sets compiler to or not to translate XcalableMP directive.
     *
     * @param enable true then translate XcalableMP directive.
     */
    public static void setIsXcalableMP(boolean enable)
    {
        _xcalableMP = enable;
    }

    /**
     * Checks does compiler translate XcalableMP directive.
     *
     * @return true if compiler translate XcalableMP directive.
     */
    public static boolean isXcalableMP()
    {
        return _xcalableMP;
    }

    /**
     * Sets compiler to or not to translate OpenMP directive.
     *
     * @param enable true then translate XcalableMP directive.
     */
    public static void setIsOpenMP(boolean enable)
    {
        _openMP = enable;
    }

    /**
     * Checks does compiler translate OpenMP directive.
     *
     * @return true if compiler translate OpenMP directive.
     */
    public static boolean isOpenMP()
    {
        return _openMP;
    }

    /**
     * Return true if debug output enabled.
     */
    public static boolean isDebugOutput()
    {
        return _debugOutput;
    }

    /**
     * Set debug output.
     */
    public static void setDebugOutput(boolean enable)
    {
        _debugOutput = enable;
    }

    /**
     * Set language
     */
    public static void setLanguage(XmLanguage lang)
    {
        _language = lang;
    }
    
    /**
     * Get language
     */
    public static XmLanguage getLanguage()
    {
        return _language;
    }

    /**
     * Return if the language is C
     */
    public static boolean isLanguageC()
    {
        return _language.equals(XmLanguage.C);
    }

    /**
     * Return if the language is Fortran
     */
    public static boolean isLanguageF()
    {
        return _language.equals(XmLanguage.F);
    }

    /**
     * Return compiler vendor constant. (COMP_VENDOR_*)
     */
    public static int getCompilerVendor()
    {
        return _compilerVendor;
    }

    /**
     * Set compiler vendor constant. (COMP_VENDOR_*)
     */
    public static void setCompilerVendor(int vendor)
    {
        _compilerVendor = vendor;
    }

    /**
     * Get if or not IO statements are transformed to atomic operation.
     */
    public static boolean isAtomicIO()
    {
        return _isAtomicIO || _compilerVendor == COMP_VENDOR_INTEL;
    }

    /**
     * Set if or not IO statements are transformed to atomic operation.
     */
    public static void setIsAtomicIO(boolean atomicIO)
    {
        _isAtomicIO = atomicIO;
    }
}
