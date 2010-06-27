/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * interface for sturct/union/enum type.
 */
public interface XcTaggedType
{
    public String getTagName();

    public String getTypeNameHeader();

    public void setTagName(String tagName);
}
