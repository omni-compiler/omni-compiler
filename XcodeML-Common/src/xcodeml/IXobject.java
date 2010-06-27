/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml;

public interface IXobject
{
    /** find() argument: find any symbol */
    public static int FINDKIND_ANY = 0;
    /** find() argument: find variable symbol */
    public static int FINDKIND_VAR = 1;
    /** find() argument: find common block symbol */
    public static int FINDKIND_COMMON = 2;
    /** find() argument: find tag, structure name symbol */
    public static int FINDKIND_TAGNAME = 3;

    /**
     * get line number.
     */
    public ILineNo getLineNo();

    /**
     * get parent IXobject.
     */
    public IXobject getParent();

    /**
     * set parent IXobject.
     */
    public void setParentRecursively(IXobject parent);

    /**
     * find symbol object.
     * 
     * @param name
     *      symbol name
     * @param kind
     *      FINDKIND_*
     */
    public IXobject find(String name, int kind);
}
