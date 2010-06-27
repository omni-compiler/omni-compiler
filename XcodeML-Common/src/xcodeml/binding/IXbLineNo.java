/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.binding;

/**
 * Represents element which has lineno and file attribute.
 */
public interface IXbLineNo
{
    /**
     * Gets a line number of original statement/expression/declaration in source code.
     *
     * @return line number description.
     */
    public String getLineno();

    public void setLineno(String lineno);

    /**
     * Gets a raw line number of the original statement/expression/declaration in pre processed source code.
     *
     * @return line number description.
     */
    public String getRawlineno();

    public void setRawlineno(String rawlineno);

    /**
     * Gets file name of the original statement/expression/declaration described.
     *
     * @return file name
     */
    public String getFile();

    public void setFile(String file);
}
