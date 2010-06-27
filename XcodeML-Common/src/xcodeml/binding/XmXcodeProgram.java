/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.binding;

/**
 * Represents XcodeProgram
 */
public interface XmXcodeProgram extends IXmlElement
{
    /**
     * Get language.
     */
    public String getLanguage();

    /**
     * Get source code path.
     */
    public String getSource();
}
