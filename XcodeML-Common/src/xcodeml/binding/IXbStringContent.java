/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.binding;

/**
 * The interface for XML tag elements those have a string content.
 */
public interface IXbStringContent
{
    /**
     * Gets a string content of the tag.
     * 
     * @return a string content of the tag.
     */
    public String getContent();

    public void setContent(String content);
}
